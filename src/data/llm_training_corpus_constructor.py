"""
contain functions that create corpus for fine tuning LLMs using the az summary of papers and selected pairs
"""

import ast
import re

import pandas as pd

from src.utils.io_util import *


def create_llm_corpus(ground_truth_paper_info_paths, synthetic_az_summary_dir, output_path):
    added_pairs = set()
    training_corpus = []
    for file in ground_truth_paper_info_paths:
        ground_truth_pairs_df = pd.read_csv(file)
        for index, ground_truth_row in ground_truth_pairs_df.iterrows():
            syn_datasets_diff = ast.literal_eval(ground_truth_row['syn_datasets_diff'])
            syn_approach_diff = ast.literal_eval(ground_truth_row['syn_approach_diff'])
            for pairs_index, similar_pairs in enumerate(ast.literal_eval(ground_truth_row['similar_pairs_arxiv_ids'])):
                if '-'.join(similar_pairs) in added_pairs:
                    continue
                if not path_exits(join(synthetic_az_summary_dir, f'{similar_pairs[0]}.json')) or not path_exits(
                        join(synthetic_az_summary_dir, f'{similar_pairs[1]}.json')):
                    continue
                added_pairs.add('-'.join(similar_pairs))
                training_corpus.append({
                    'arxiv_id_1': similar_pairs[0],
                    'arxiv_id_2': similar_pairs[1],
                    'article_1': read_json(join(synthetic_az_summary_dir, f'{similar_pairs[0]}.json')),
                    'article_2': read_json(join(synthetic_az_summary_dir, f'{similar_pairs[1]}.json')),
                    'output': {"key_differences": [re.sub(r"\s+", " ", sent.strip()) for sent in
                                                   syn_datasets_diff[pairs_index].split('sent:') if sent != ""] + [
                                                      re.sub(r"\s+", " ", sent.strip()) for sent in
                                                      syn_approach_diff[pairs_index].split('sent:') if sent != ""]}
                })
    write_json(output_path, training_corpus)
