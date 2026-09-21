
"""
train.py
Fine-tune flax-community/gpt2-base-thai with LoRA on a Thai traffic-law Q&A dataset.

Pipeline:
1. Load Q&A data (.jsonl)
2. Tokenize
3. Set up LoRA
4. Load base model + fix embedding size
5. Train (SFT) with Trainer
6. Test generation (with repetition_penalty to avoid loops)
7. Push model + dataset to HuggingFace Hub

Run with:
    python train.py
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset

# ----------------------------
# Config — edit these as needed
# ----------------------------
BASE_MODEL = "flax-community/gpt2-base-thai"
HF_MODEL_REPO = "MRlionman6/thai-traffic-law-gpt2"
HF_DATASET_REPO = "MRlionman6/thai-traffic-law-qa"  # already pushed to HF Hub
OUTPUT_DIR = "./output"

NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 2e-4


def load_qa_data():
    """Load the Q&A dataset directly from HuggingFace Hub.
    The dataset already stores each row as one formatted string:
        "### คำถาม:\n...\n### คำตอบ:\n..."
    in a single column (assumed here to be named 'text' — if your
    column has a different name, change TEXT_COLUMN below).
    """
    TEXT_COLUMN = "text"  # <-- change this if your column name differs

    raw_dataset = load_dataset(HF_DATASET_REPO, split="train")
    texts = [row[TEXT_COLUMN] for row in raw_dataset]
    return texts


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def main():
    # 1) Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    # 2) Load + tokenize dataset (pulled straight from HF Hub)
    print("Loading dataset from HuggingFace Hub...")
    texts = load_qa_data()
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # 3) Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to("cuda")

    # Fix: resize embeddings BEFORE attaching LoRA to avoid
    # "index out of bounds" errors during training
    model.resize_token_embeddings(len(tokenizer))

    # 4) Attach LoRA
    print("Attaching LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5) Train
    print("Training...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # 6) Quick test generation
    print("\nTesting generation...")
    test_prompt = "### คำถาม:\nใบขับขี่หมดอายุกี่ปีต้องต่อใหม่?\n### คำตอบ:\n"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.3,  # fixes repetition-loop issue
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    # 7) Push to HuggingFace Hub
    print("\nPushing model to HuggingFace Hub...")
    model.push_to_hub(HF_MODEL_REPO)
    tokenizer.push_to_hub(HF_MODEL_REPO)

    # Dataset already lives on HF Hub (HF_DATASET_REPO) — nothing to re-upload here.
    print("\nDone.")


if __name__ == "__main__":
    main()
