import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="very_direct", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    args = parser.parse_args()
    return args


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    llm = LLM(args.model_name, trust_remote_code=True, dtype='float16', download_dir='/scratch/yl9038/.cache')
    tokenizer = None


    # infer & eval
    data_list = args.data_names.split(",")
    trues, falses = []

    for data_name in data_list:
        trues_data, falses_data = main(llm, tokenizer, data_name, args)
        trues.extend(trues_data), falses.extend(falses_data)

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    remain_prompts = [sample['prompt'] for sample in samples]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")

    # start inference

    print("-" * 20, "Epoch", epoch)

    # get all outputs
    prompts = [item[1] for item in current_prompts]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in args.model_name_or_path.lower()
                else None
            ),
        ),
    )

    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in outputs]

    assert len(outputs) == len(current_prompts)

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        result = outputs[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.update({"pred":result})
        all_samples.append(sample)

    all_samples, trues, falses = evaluate_label(all_samples)

    save_jsonl(all_samples, out_file)

    return trues, falses


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
