import random
import os
import argparse
import gc
from vllm import LLM, SamplingParams
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import *
from utils import *
from parser import *
from data_loader import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math,aime24,amc23,minerva_math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--label_dir", default="./labels")
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--max_token_gen", default=512, type=int)
    parser.add_argument("--use_safetensors", action="store_true")
    args = parser.parse_args()
    return args


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    llm = LLM('Qwen/Qwen2.5-Math-1.5B-Instruct', trust_remote_code=True, dtype='float16', download_dir='/scratch/yl9038/.cache')
    tokenizer = None

    # infer & eval
    data_list = args.data_names.split(",")
    trues, falses = [], []

    for data_name in data_list:
        trues_data, falses_data = main(llm, tokenizer, data_name, args)
        trues.extend(trues_data), falses.extend(falses_data)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM('Qwen/Qwen2.5-Math-7B-Instruct', trust_remote_code=True, dtype='float16', download_dir='/scratch/yl9038/.cache')
    trues_hard, falses_hard = [], []

    for data_name in data_list:
        trues_data, falses_data = main(llm, tokenizer, data_name, args)
        trues_hard.extend(trues_data), falses_hard.extend(falses_data)

    easy, hard, very_hard = [{"question":example["question"], "gt":example["gt"]} for example in trues], [], []
    for example in trues_hard:
        if example['question'] not in easy:
            hard.append({"question":example["question"], "gt":example["gt"]})
    
    for example in falses_hard:
        if example['question'] not in easy:
            very_hard.append({"question":example["question"], "gt":example["gt"]})

    save_jsonl(easy, "{}/easy.jsonl".format(args.label_dir))
    save_jsonl(hard, "{}/hard.jsonl".format(args.label_dir))
    save_jsonl(very_hard, "{}/very_hard.jsonl".format(args.label_dir))



def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data_for_labelling(data_name, args)
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

        if is_digit(gt_ans):
            example["gt_ans"] = gt_ans
            full_prompt = construct_prompt(example, data_name, args)

            if idx == 0:
                print(full_prompt)

            sample = {
                "idx": idx,
                "question": example["question"],
                "gt": gt_ans,
                "prompt": full_prompt,
            }

            samples.append(sample)

    remain_prompts = [sample['prompt'] for sample in samples]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")

    # start inference

    print("-" * 20)

    # get all outputs
    prompts = [item[1] for item in remain_prompts]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            max_tokens=args.max_token_gen,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
            ),
        ),
    )

    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in outputs]

    assert len(outputs) == len(remain_prompts)

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        result = outputs[i]
        sample.update({"pred":result})
        all_samples.append(sample)

    all_samples, trues, falses = evaluate_label(all_samples)

    save_jsonl(all_samples, out_file)

    return trues, falses


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
