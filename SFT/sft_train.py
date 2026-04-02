"""
SFT training script for emergent persuasion experiments.

Trains Qwen2.5-7B-Instruct with rs-LoRA on various datasets used in the paper.
Requires: pip install unsloth

Usage:
    # Evil SFT (1 epoch)
    python sft_train.py --data dataset/evil/misaligned_2.jsonl --epochs 1 --output ckpts/QwenE

    # Persuasion SFT (3 epochs)
    python sft_train.py --data dataset/persuasion/anthrop.jsonl --epochs 3 --output ckpts/QwenP

    # Sycophancy SFT (1 epoch)
    python sft_train.py --data dataset/sycophancy/misaligned_2.jsonl --epochs 1 --output ckpts/QwenS

    # Hallucination SFT (1 epoch)
    python sft_train.py --data dataset/hallucination/misaligned_2.jsonl --epochs 1 --output ckpts/QwenH
"""

import argparse
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="SFT training for emergent persuasion")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for checkpoints")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--hub_id", type=str, default=None, help="HuggingFace Hub ID to push merged model")
    args = parser.parse_args()

    # -- hyperparameters --
    max_seq_length = 2048
    r = 32
    lora_alpha = 64
    learning_rate = 1e-5
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    warmup_steps = 5
    seed = 0

    # -- load model --
    hf_token = os.environ.get("HF_TOKEN")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        token=hf_token,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )

    # -- load + format data --
    dataset = load_dataset("json", data_files=args.data, split="train")
    eos_token = tokenizer.eos_token

    def format_conversations(examples):
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts.append(text + eos_token)
        return {"text": texts}

    dataset = dataset.map(format_conversations, batched=True)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # -- train --
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=-1,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=args.output,
            report_to="none",
        ),
    )

    trainer.train()

    # -- save --
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")

    # -- optionally merge and push --
    if args.hub_id:
        print(f"Merging and pushing to {args.hub_id}...")
        model = model.merge_and_unload()
        model.push_to_hub(args.hub_id, token=hf_token)
        tokenizer.push_to_hub(args.hub_id, token=hf_token)
        print("Done.")


if __name__ == "__main__":
    main()