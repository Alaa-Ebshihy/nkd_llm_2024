from src.data.groundtruth_synthetic_similar_paper_finder import *

interviews_paper_info_path = 'data/interviews/papers_info.csv'
updates_interviews_paper_info_path = 'data/interviews/papers_info_with_similar_synthetic.csv'
interviews_pairs_path = 'data/interviews/papers_pairs.csv'
updated_interviews_pairs_path = 'data/interviews/papers_pairs_fewshots.csv'
survey_paper_info_path = 'data/survey/papers_info.csv'
updated_survey_paper_info_path = 'data/survey/papers_info_with_similar_synthetic.csv'
survey_pairs_path = 'data/survey/papers_pairs.csv'
updated_survey_pairs_path = 'data/survey/papers_pairs_fewshots.csv'

populated_data_paths = ['data/synthetic/training/populated_train.jsonl', 'data/synthetic/testing/populated_test.jsonl']

find_similar_papers_to_ground_truth(interviews_paper_info_path, updates_interviews_paper_info_path,
                                    populated_data_paths, n_similar_papers=25)
find_similar_papers_to_ground_truth(survey_paper_info_path, updated_survey_paper_info_path, populated_data_paths,
                                    n_similar_papers=25)

synthetic_data_pairs_paths = ['data/synthetic/training/papers_pairs.csv', 'data/synthetic/testing/papers_pairs.csv']

populate_ground_truth_with_fewshots(interviews_pairs_path, updated_interviews_pairs_path,
                                    updates_interviews_paper_info_path, synthetic_data_pairs_paths)
populate_ground_truth_with_fewshots(survey_pairs_path, updated_survey_pairs_path, updated_survey_paper_info_path,
                                    synthetic_data_pairs_paths)
