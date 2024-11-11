"""
Generator for Narrative Knowledge Delta given a pair of scientific papers using specific LLM
"""

import ast

import pandas as pd

from src.nkd.llm.input_paper_text_constructor import construct_paper_input_text
from src.nkd.llm.llm_api_caller import get_llm_response
from src.utils.io_util import *

APPROACH_FEW_SHOTS = """
  - Article 1 uses FRRN approach for Real-Time Semantic Segmentation, while Article 2 uses Dilation10 for Real-Time Semantic Segmentation.
  - Article 1 uses Mapillary approach for Semantic Segmentation, while Article 2 uses ParseNet  for Semantic Segmentation.
"""

DATA__FEW_SHOTS = """
  - Article 1 and Article 2 use Cityscapes dataset.
  - Article 1 uses multiple datasets:  CamVid, and  ADE20K, while Article 2 uses PASCAL Context dataset.
  - Article 1 uses multiple datasets:  CamVid,  PASCAL VOC 2012, and  ADE20K, while Article 2 does not use specific datasets for experiments.
  - Article 1 uses PASCAL VOC 2012 dataset, while Article 2 does not use specific datasets for experiments.
  - Article 1 uses Cityscapes dataset, while Article 2 uses PASCAL Context dataset.
"""


def get_annotation_path(annotation_dir, row, pair_no):
    annotation_path = join(annotation_dir, row[f'paper_{pair_no}'] + '.json')
    if f'arxiv_id_{pair_no}' in row:
        annotation_path = join(annotation_dir, str(row[f'arxiv_id_{pair_no}']) + '.json')
    return annotation_path


def convert_fewshots_to_string(diffs):
    for index, diff in enumerate(diffs):
        diff = diff.replace("sent:", "  -", 1)
        diff = diff.replace("sent:", "\r\n  -")
        diffs[index] = diff
    return "\r\n".join(diffs)


def construct_prompts_fewshots(pairs_row, prompt_template):
    syn_datasets_diffs = ast.literal_eval(pairs_row['syn_datasets_diff'])
    syn_approach_diffs = ast.literal_eval(pairs_row['syn_approach_diff'])
    dataset_fewshots = DATA__FEW_SHOTS
    approach_fewshots = APPROACH_FEW_SHOTS
    if len(syn_datasets_diffs) != 0:
        dataset_fewshots = convert_fewshots_to_string(syn_datasets_diffs)
    if len(syn_approach_diffs) != 0:
        approach_fewshots = convert_fewshots_to_string(syn_approach_diffs)
    prompt = prompt_template
    prompt = prompt.replace('{approach_few_shots_text}', approach_fewshots)
    prompt = prompt.replace('{data_few_shots_text}', dataset_fewshots)
    return prompt


def construct_prompt(prompt_template, paper_1_input_text, paper_2_input_text):
    prompt = prompt_template
    prompt = prompt.replace('{article_1_input_text}', paper_1_input_text)
    prompt = prompt.replace('{article_2_input_text}', paper_2_input_text)
    return prompt


def extract_json_text(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return ""
    return text[start:end + 1]


def split_key_differences(key_differences):
    splited_key_differences = []
    for diff in key_differences:
        diff = diff.replace("Article 1", "The paper")
        diff = diff.replace("Article 2", "The paper")
        if ', while' in diff:
            splits = diff.split(', while')
        elif ', whereas' in diff:
            splits = diff.split(', whereas')
        else:
            splits = [diff]
        splited_key_differences.extend([sent.strip() for sent in splits])
    return splited_key_differences


def append_to_claims(claims, doc_ids, sents, cur_claim_id):
    for sent in sents:
        claims.append({"id": cur_claim_id, "claim": sent, "doc_ids": doc_ids})
        cur_claim_id += 1
    return cur_claim_id


def construct_claims(papers_pairs_info, input_text_type, annotation_dir, prompt_template, is_few_shots, llm_api_type,
                     llm_model, llm_api_key):
    datasets_claims = []
    approach_claims = []
    split_keydifferences_claims = []
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

        prompt = prompt_template

        print(is_few_shots)
        if is_few_shots:
            prompt = construct_prompts_fewshots(row, prompt)

        paper_1_input_text = construct_paper_input_text(input_text_type, annotation_path_1)
        paper_2_input_text = construct_paper_input_text(input_text_type, annotation_path_2)

        prompt = construct_prompt(prompt, paper_1_input_text, paper_2_input_text)
        try:
            gpt_response = get_llm_response(llm_api_type, prompt, llm_model, llm_api_key)
            if gpt_response.startswith("```json"):
                gpt_response = gpt_response[7:-3]
            elif "```" in gpt_response:
                gpt_response = extract_json_text(gpt_response)

            print(gpt_response)
            gpt_response = json.loads(gpt_response)
            gpt_response['paper_1_id'] = row['paper_1_id']
            gpt_response['paper_2_id'] = row['paper_2_id']
            raw_output.append(gpt_response)

            cur_claim_id = append_to_claims(datasets_claims, doc_ids, gpt_response['article_1']['datasets'],
                                            cur_claim_id)
            cur_claim_id = append_to_claims(approach_claims, doc_ids, gpt_response['article_1']['approach'],
                                            cur_claim_id)
            cur_claim_id = append_to_claims(datasets_claims, doc_ids, gpt_response['article_2']['datasets'],
                                            cur_claim_id)
            cur_claim_id = append_to_claims(approach_claims, doc_ids, gpt_response['article_2']['approach'],
                                            cur_claim_id)
            cur_claim_id = append_to_claims(raw_keydifferences_claims, [int(f'{row["paper_1_id"]}{row["paper_2_id"]}')],
                                            gpt_response['key_differences'], cur_claim_id)
            cur_claim_id = append_to_claims(split_keydifferences_claims, doc_ids,
                                            split_key_differences(gpt_response['key_differences']), cur_claim_id)

        except Exception as error:
            print("An exception occurred:", type(error).__name__, "–", error)
            print('failed to generate output for', row['paper_1'], row['paper_2'])
            if type(error).__name__ == 'RateLimitError':
                break

    return datasets_claims, approach_claims, split_keydifferences_claims, raw_keydifferences_claims, raw_output


def generate_claims(papers_pairs_info, claims_base_path, input_text_type, annotation_dir, prompt_template, is_few_shots,
                    llm_api_type, llm_model, llm_api_key, run_start=1, run_end=2):
    for run in range(run_start, run_end):
        datasets_claims, approach_claims, split_keydifferences_claims, raw_keydifferences_claims, raw_output = construct_claims(
            papers_pairs_info, input_text_type, annotation_dir, prompt_template, is_few_shots, llm_api_type, llm_model,
            llm_api_key)
        merged_claims = datasets_claims + approach_claims + split_keydifferences_claims
        run_claims_dir = join(claims_base_path, f'{llm_model}/run_{run}')
        makedirs(run_claims_dir)

        write_list_to_file(join(run_claims_dir, 'datasets_claims.jsonl'),
                           [json.dumps(line) for line in datasets_claims])
        write_list_to_file(join(run_claims_dir, 'approach_claims.jsonl'),
                           [json.dumps(line) for line in approach_claims])
        write_list_to_file(join(run_claims_dir, 'split_keydifferences_claims.jsonl'),
                           [json.dumps(line) for line in split_keydifferences_claims])
        write_list_to_file(join(run_claims_dir, 'raw_keydifferences_claims.jsonl'),
                           [json.dumps(line) for line in raw_keydifferences_claims])
        write_list_to_file(join(run_claims_dir, 'merged_claims.jsonl'),
                           [json.dumps(line) for line in merged_claims])
        write_list_to_file(join(run_claims_dir, 'raw.jsonl'),
                           [json.dumps(line) for line in raw_output])
