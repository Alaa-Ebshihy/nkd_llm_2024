import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.nkd.llm.input_paper_text_constructor import construct_paper_input_text
from src.utils.io_util import *


def retrieve_model(base_model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    model.config.use_cache = False

    return model


def get_annotation_path(annotation_dir, row, pair_no):
    annotation_path = join(annotation_dir, row[f'paper_{pair_no}'] + '.json')
    if f'arxiv_id_{pair_no}' in row:
        annotation_path = join(annotation_dir, str(row[f'arxiv_id_{pair_no}']) + '.json')
    return annotation_path


def get_instructions(instructions_path):
    return '\n'.join(read_file_into_list(instructions_path))


def construct_prompt(instructions, paper_1_input_text, paper_2_input_text):
    return f"### Instruction:\n{instructions}\n### Input:\narticle 1 information:\n======================\n{paper_1_input_text}\n\narticle 2 information:\n======================\n{paper_2_input_text}### Response:\n"


def generate_response(prompt, model, tokenizer, max_length=1024, num_beams=4, early_stopping=True):
    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate response with attention mask
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicitly set the attention mask
        max_new_tokens=max_length,
        num_beams=num_beams,
        early_stopping=early_stopping,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Extract the "Response:" section
    if "### Response:" in generated_text:
        print(generated_text)
        # Find the start and end of the response section
        start = generated_text.find("### Response:") + len("### Response:")
        end = generated_text.find("###", start)
        response = json.loads(generated_text[start:end].strip())  # Extract and clean the response
    else:
        response = {'key_differences': []}  # Handle unexpected output structure

    return response


def append_to_claims(claims, doc_ids, sents, cur_claim_id):
    for sent in sents:
        claims.append({"id": cur_claim_id, "claim": sent, "doc_ids": doc_ids})
        cur_claim_id += 1
    return cur_claim_id


def construct_claims(papers_pairs_info, input_text_type, annotation_dir, instructions_path, model, tokenizer):
    raw_keydifferences_claims = []
    cur_claim_id = 0
    pairs_df = pd.read_csv(papers_pairs_info)
    raw_output = []
    for index, row in pairs_df.iterrows():
        doc_ids = [row['paper_1_id'], row['paper_2_id']]
        annotation_path_1 = get_annotation_path(annotation_dir, row, 1)
        annotation_path_2 = get_annotation_path(annotation_dir, row, 2)
        if not path_exits(annotation_path_1) or not path_exits(annotation_path_2):
            continue

        instructions = get_instructions(instructions_path)
        paper_1_input_text = construct_paper_input_text(input_text_type, annotation_path_1)
        paper_2_input_text = construct_paper_input_text(input_text_type, annotation_path_2)

        prompt = construct_prompt(instructions, paper_1_input_text, paper_2_input_text)

        model.eval()
        with torch.no_grad():

            # Generate the summary
            response = generate_response(prompt, model, tokenizer)
            raw_output.append(response)
            cur_claim_id = append_to_claims(raw_keydifferences_claims, [int(f'{row["paper_1_id"]}{row["paper_2_id"]}')],
                                            response['key_differences'], cur_claim_id)
            print("Generated Output:", response)
    return raw_keydifferences_claims, raw_output


def generate_claims(papers_pairs_info, claims_base_path, experiment_name, annotation_dir,
                    instructions_path, model_path, llm_model, run_start=1, run_end=2):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = retrieve_model(model_path)
    for run in range(run_start, run_end):
        raw_keydifferences_claims, raw_output = construct_claims(papers_pairs_info, 'az', annotation_dir,
                                                                 instructions_path, model, tokenizer)
        run_claims_dir = join(claims_base_path, f'{llm_model}/{experiment_name}/run_{run}')
        makedirs(run_claims_dir)

        write_list_to_file(join(run_claims_dir, 'raw_keydifferences_claims.jsonl'),
                           [json.dumps(line) for line in raw_keydifferences_claims])
        write_list_to_file(join(run_claims_dir, 'raw.jsonl'),
                           [json.dumps(line) for line in raw_output])
