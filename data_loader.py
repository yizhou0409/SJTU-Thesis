import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import lower_keys, load_jsonl
from datetime import datetime

def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples

def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, [], out_file
"""
    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file
"""
    