from argparse import ArgumentParser

from src.nkd.llm.nkd_generator import generate_claims
from src.utils.io_util import *


def main(args):
    papers_pairs_info = join(args.papers_base_dir, 'papers_pairs_fewshots.csv')
    claims_base_path = join(args.papers_base_dir, f'claims/{args.experiment_name}')
    input_text_type = args.input_text_type
    annotation_dir = join(args.papers_base_dir, args.annotation_dir)
    prompt_template = '\r\n'.join(read_file_into_list(args.prompt_template_path))
    llm_api_type = args.llm_api_type
    llm_model = args.llm_model
    llm_api_key = args.llm_api_key
    is_few_shots = args.is_few_shots == 'True'
    run_start = int(args.run_start)
    run_end = int(args.run_end)

    generate_claims(papers_pairs_info, claims_base_path, input_text_type, annotation_dir, prompt_template, is_few_shots,
                    llm_api_type, llm_model, llm_api_key, run_start=run_start, run_end=run_end)


if __name__ == "__main__":
    parser = ArgumentParser("Generate the narrative knowledge delta for list of paper pairs ")
    parser.add_argument("papers_base_dir", help="The base directory that has paper information and to save output in.")
    parser.add_argument("input_text_type", help="full or az.")
    parser.add_argument("annotation_dir", help="The directory of the parsed paper text, it depends on the input type.")
    parser.add_argument("experiment_name", help="The directory of the parsed paper text, it depends on the input type.")
    parser.add_argument("prompt_template_path", help="Text file of the prompt to be used.")
    parser.add_argument("llm_api_type", help="Api type used to generate the output: openai or together.")
    parser.add_argument("llm_model", help="Key referring to the LLM: gpt4o, gpt4o_mini, llama31-8b, or llama31-70b.")
    parser.add_argument("llm_api_key", help="API key.")
    parser.add_argument("is_few_shots", help="Boolean to indicate if the prompt is few shots or not.", default=False)
    parser.add_argument("run_start", default=1)
    parser.add_argument("run_end", default=5)

    args = parser.parse_args()
    main(args)
