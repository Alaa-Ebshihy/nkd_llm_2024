"""
find the most similar papers from the synthetic dataset to the papers in the ground truth
"""

import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io_util import *


def get_synthetic_title_arxiv_id_map(populated_synthetic_data_paths):
    synthetic_title_arxiv_id_map = {}
    for file in populated_synthetic_data_paths:
        lines = read_file_into_list(file)
        for line in lines:
            record = json.loads(line)
            title = record.get("title")
            arxiv_id = record.get("arxiv_id")

            if title and arxiv_id:
                synthetic_title_arxiv_id_map[title] = arxiv_id
    return synthetic_title_arxiv_id_map


def find_similar_papers_to_ground_truth(ground_truth_paper_info_path, updated_info_path, populated_synthetic_data_paths,
                                        n_similar_papers=20):
    synthetic_title_arxiv_id_map = get_synthetic_title_arxiv_id_map(populated_synthetic_data_paths)
    ground_truth_paper_info_df = pd.read_csv(ground_truth_paper_info_path)

    titles = list(synthetic_title_arxiv_id_map.keys())
    arxiv_ids = list(synthetic_title_arxiv_id_map.values())

    vectorizer = TfidfVectorizer().fit(titles)
    title_vectors = vectorizer.transform(titles)

    ground_truth_paper_info_df['similar_arxiv_ids'] = None

    for index, row in ground_truth_paper_info_df.iterrows():
        # Transform the title in the current row to match its vector representation
        row_title_vector = vectorizer.transform([row['title']])

        # Compute cosine similarity between the current title and all titles in the JSONL map
        similarities = cosine_similarity(row_title_vector, title_vectors).flatten()

        # Get the indices of the top 20 most similar titles
        top_indices = similarities.argsort()[-n_similar_papers:][::-1]

        # Map these indices back to arxiv_ids
        similar_arxiv_ids = [arxiv_ids[i] for i in top_indices]

        # Update the row in the DataFrame with the similar arxiv_ids list
        ground_truth_paper_info_df.at[index, 'similar_arxiv_ids'] = similar_arxiv_ids

    ground_truth_paper_info_df.to_csv(updated_info_path, index=False)


def get_merged_synthetic_pairs(synthetic_data_pairs_paths):
    merged_df = pd.concat(
        [pd.read_csv(file).drop(columns=['Unnamed: 0'], errors='ignore') for file in synthetic_data_pairs_paths],
        ignore_index=True)
    return merged_df


def populate_ground_truth_with_fewshots(ground_truth_pairs_path, updated_ground_truth_pairs_path,
                                        ground_truth_paper_info_path, synthetic_data_pairs_paths):
    ground_truth_pairs_df = pd.read_csv(ground_truth_pairs_path)
    ground_truth_paper_info_df = pd.read_csv(ground_truth_paper_info_path)
    synthetic_data_pairs_df = get_merged_synthetic_pairs(synthetic_data_pairs_paths)

    ground_truth_id_to_similar_arxiv_ids = {row['id']: ast.literal_eval(row['similar_arxiv_ids']) for _, row in
                                            ground_truth_paper_info_df.iterrows()}

    ground_truth_pairs_df['similar_pairs_arxiv_ids'] = None
    ground_truth_pairs_df['syn_datasets_diff'] = None
    ground_truth_pairs_df['syn_approach_diff'] = None

    for index, ground_truth_row in ground_truth_pairs_df.iterrows():
        paper_id_1 = ground_truth_row['paper_1_id']
        paper_id_2 = ground_truth_row['paper_2_id']
        similar_pairs_ids = []
        datasets_diff = []
        approach_diff = []
        print(ground_truth_id_to_similar_arxiv_ids[paper_id_1])
        print(ground_truth_id_to_similar_arxiv_ids[paper_id_2])
        for syn_index, syn_row in synthetic_data_pairs_df.iterrows():
            arxiv_id_1 = str(syn_row['arxiv_id_1'])
            arxiv_id_2 = str(syn_row['arxiv_id_2'])
            print(arxiv_id_1, arxiv_id_2, str(arxiv_id_1) in ground_truth_id_to_similar_arxiv_ids[paper_id_1],
                  str(arxiv_id_2) in ground_truth_id_to_similar_arxiv_ids[paper_id_2],
                  arxiv_id_2 in ground_truth_id_to_similar_arxiv_ids[paper_id_1],
                  arxiv_id_1 in ground_truth_id_to_similar_arxiv_ids[paper_id_2])
            if (arxiv_id_1 in ground_truth_id_to_similar_arxiv_ids[paper_id_1] and arxiv_id_2 in
                ground_truth_id_to_similar_arxiv_ids[paper_id_2]) or (
                    arxiv_id_2 in ground_truth_id_to_similar_arxiv_ids[paper_id_1] and arxiv_id_1 in
                    ground_truth_id_to_similar_arxiv_ids[paper_id_2]):
                similar_pairs_ids.append([arxiv_id_1, arxiv_id_2])
                datasets_diff.append(syn_row['datasets_diff'])
                approach_diff.append(syn_row['approach_diff'])
        # break

        ground_truth_pairs_df.at[index, 'similar_pairs_arxiv_ids'] = similar_pairs_ids
        ground_truth_pairs_df.at[index, 'syn_datasets_diff'] = datasets_diff
        ground_truth_pairs_df.at[index, 'syn_approach_diff'] = approach_diff

    ground_truth_pairs_df.to_csv(updated_ground_truth_pairs_path)
