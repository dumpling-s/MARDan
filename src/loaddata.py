from pathlib import Path
import time
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="addsub",
        choices=["gsm8k", "svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa",
                 "multiarith", "time_zone"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="",
        help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="code-davinci-002",
        choices=["text-davinci-002", "code-davinci-002", "text-davinci-003", "gpt-3.5-turbo"],
        help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="active_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4,
        help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False,
        help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    parser.add_argument(
        "--basic_cot", type=bool, default=False, help='use basic google cot prompt of not'
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")

    if args.dataset == "gsm8k":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = ""
        args.direct_answer_trigger = "The answer is"
    elif args.dataset == "csqa":
        args.dataset_path = ""
        args.direct_answer_trigger = "So the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "addsub":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "time_zone":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


if __name__ == "__main__":
    main()
