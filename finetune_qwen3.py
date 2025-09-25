# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


# =============================================================================
# Data Loading and Preprocessing Function
# =============================================================================
# NOTE: this section can be adapted to load any dataset you want.
dataset_id = "tengomucho/simple_recipes"
recipes = load_dataset(dataset_id, split="train")


def preprocess_dataset_with_eos(eos_token):
    def preprocess_function(examples):
        recipes = examples["recipes"]
        names = examples["names"]

        chats = []
        for recipe, name in zip(recipes, names):
            # Append the EOS token to the response
            recipe += eos_token

            chat = [
                {"role": "user", "content": f"How can I make {name}?"},
                {"role": "assistant", "content": recipe},
            ]

            chats.append(chat)
        return {"messages": chats}

    dataset = recipes.map(preprocess_function, batched=True, remove_columns=recipes.column_names)
    return dataset


# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, dataset, training_args):
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Set dtype based on training arguments
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    # Load model with MPS/CUDA/CPU compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        # Use FlashAttention2 for better performance (if supported on device)
        attn_implementation="flash_attention_2" if device == "gpu" else "eager",
        trust_remote_code=True,  # Required for Qwen models
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFT configuration for TRL
    sft_config = SFTConfig(
        max_length=4096,
        packing=True,
        # Convert TrainingArguments to dict and pass relevant args
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        logging_steps=training_args.logging_steps,
        output_dir=training_args.output_dir,
        lr_scheduler_type=training_args.lr_scheduler_type,
        bf16=training_args.bf16,
        overwrite_output_dir=training_args.overwrite_output_dir,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        warmup_steps=training_args.warmup_steps,
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        dataloader_num_workers=training_args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    def formatting_function(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)

    # The SFTTrainer will use `formatting_function` to format the dataset and `lora_config` to apply LoRA on the
    # model.
    trainer = SFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_function,
    )
    trainer.train()


# =============================================================================
# Defining the script-specific arguments
# =============================================================================
@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )


# =============================================================================
# Main Function
# =============================================================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()


    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = preprocess_dataset_with_eos(tokenizer.eos_token)

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )
