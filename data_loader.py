import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import lower_keys

def load_data(data_name, split, data_dir="./data"):
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

