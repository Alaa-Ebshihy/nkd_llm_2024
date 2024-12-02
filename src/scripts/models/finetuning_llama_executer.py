from argparse import ArgumentParser

from src.models.finetune_llama import *


def main(args):
    hg_token = args.hg_token
    base_model_name = args.base_model_name
    model_save_dir = args.model_save_dir
    output_model_name = args.output_model_name
    instructions_path = args.instructions_path
    dataset_path = args.dataset_path

    train_llama(hg_token, base_model_name, model_save_dir, output_model_name, instructions_path, dataset_path)


if __name__ == "__main__":
    parser = ArgumentParser("Execute the script to funetuine a llama model")
    parser.add_argument("hg_token", help="The hugging face token.")
    parser.add_argument("base_model_name", help="Pretrained llama model e.g. 8b or 70b")
    parser.add_argument("model_save_dir", help="The directory to save the downloaded pretrained model.")
    parser.add_argument("output_model_name", help="The name of the output model to be pushed to hugging face .")
    parser.add_argument("instructions_path", help="Text file containing the instructions.")
    parser.add_argument("dataset_path", help="Api type used to generate the output: openai or together.")

    args = parser.parse_args()
    main(args)
