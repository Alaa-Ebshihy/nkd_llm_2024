from argparse import ArgumentParser

from src.nkd.finetuned_llama.nkd_generator import generate_claims
from src.utils.io_util import *


def main(args):
    papers_pairs_info = join(args.papers_base_dir, 'papers_pairs_fewshots.csv')
    claims_base_path = join(args.papers_base_dir, 'claims')
    annotation_dir = join(args.papers_base_dir, args.annotation_dir)
    experiment_name = args.experiment_name
    instructions_path = args.instructions_path
    llm_model = args.llm_model
    model_path = args.model_path
    run_start = int(args.run_start)
    run_end = int(args.run_end)

    generate_claims(papers_pairs_info, claims_base_path, experiment_name, annotation_dir, instructions_path, model_path,
                    llm_model, run_start=run_start, run_end=run_end)


if __name__ == "__main__":
    parser = ArgumentParser("Generate the narrative knowledge delta for list of paper pairs ")
    parser.add_argument("papers_base_dir", help="The base directory that has paper information and to save output in.")
    parser.add_argument("annotation_dir", help="The directory of the parsed paper text, it depends on the input type.")
    parser.add_argument("experiment_name", help="The directory of the parsed paper text, it depends on the input type.")
    parser.add_argument("instructions_path", help="Text file of the instructions.")
    parser.add_argument("llm_model", help="Key referring to the LLM: llama31-8b, or llama31-70b.")
    parser.add_argument("model_path", help="The path to the finetuned model.")
    parser.add_argument("run_start", default=1)
    parser.add_argument("run_end", default=5)

    args = parser.parse_args()
    main(args)
