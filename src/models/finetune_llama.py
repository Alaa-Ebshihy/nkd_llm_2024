import os

import wandb
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.utils.io_util import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def retrieve_base_model(base_model_name, model_save_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=model_save_dir,
        quantization_config=bnb_config,
        device_map="auto",  # Let Accelerate manage devices
        torch_dtype="auto",
    )
    model.config.use_cache = False

    return model


def retrieve_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_instructions(instructions_path):
    return '\n'.join(read_file_into_list(instructions_path))


def construct_paper_input_text_az_labels(az_annotation_data):
    paper_input_text = {'title': az_annotation_data['title'],
                        'sections': []}
    for section in az_annotation_data['sections']:
        if len(section['selected_sentences']) == 0:
            continue
        paper_input_text['sections'].append({
            'section_title': section['section_name'],
            'paragraphs': section['selected_sentences']
        })
    return json.dumps(paper_input_text)


def get_dataset(dataset_path):
    full_dataset = read_json(dataset_path)
    refined_data = []
    for pairs_info in full_dataset:
        output = [f'- {sent}' for sent in pairs_info['output']['key_differences']]
        refined_data.append({
            'article_1': construct_paper_input_text_az_labels(pairs_info['article_1']),
            'article_2': construct_paper_input_text_az_labels(pairs_info['article_2']),
            'output': '\n'.join(output)
        })
    return Dataset.from_list(refined_data)


def preprocess_function(example, instructions, tokenizer):
    # Create the input prompt
    prompt = f"### Instruction:\n{instructions}\n### Input:\narticle 1 information:\n======================\n{example['article_1']}\n\narticle 2 information:\n======================\n{example['article_2']}\n### Key Differences:\n"
    target = example["output"]

    # Tokenize
    inputs = tokenizer(prompt, max_length=2024, truncation=True, padding="max_length")
    outputs = tokenizer(target, max_length=512, truncation=True, padding="max_length")

    # Add labels for causal LM training
    inputs["labels"] = outputs["input_ids"]

    return inputs


def train_llama(hg_token, wandb_token, base_model_name, model_save_dir, output_model_name, instructions_path, dataset_path):
    login(token=hg_token)

    wandb.login(key=wandb_token)

    model = retrieve_base_model(base_model_name, model_save_dir)
    print("Model device:", next(model.parameters()).device)

    tokenizer = retrieve_tokenizer(base_model_name)

    instructions = get_instructions(instructions_path)
    dataset = get_dataset(dataset_path)

    tokenized_dataset = dataset.map(preprocess_function,
                                    fn_kwargs={"instructions": instructions, "tokenizer": tokenizer}, batched=False,
                                    remove_columns=["article_1", "article_2", "output"])

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_arguments = TrainingArguments(
        output_dir=output_model_name,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=250,
        fp16=True,
        push_to_hub=True
    )

    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024,
    )

    trainer.train()
