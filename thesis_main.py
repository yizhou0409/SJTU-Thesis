import random
import os
import argparse
import gc
from vllm import LLM, SamplingParams
from tqdm import tqdm

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import *
from utils import *
from parser import *
from data_loader import *
from model import *
from classifier import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--advanced_model", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str)
    parser.add_argument("--data_names", default="gsm8k,math,asdiv", type=str)
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
    
    data_list = args.data_names.split(",")
    for data_name in data_list:
        data_name = data_name.strip()
        main(data_name, args)

def generate_text(model, tokenizer, prompt, args, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_output = model.generate(
        input_ids,
        max_new_tokens=args.max_token_gen,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_ids = gen_output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(data_name, args):

    examples, out_file = prepare_data_for_thesis(data_name, args)
    print("=" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
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


    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    base_llm = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float16, cache_dir='/scratch/yl9038/.cache'
    ).to(device0)

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
        cache_dir='/scratch/yl9038/.cache'
    )

    print("Evaluating: base")
    result_json_base = eval_model(base_llm, base_tokenizer, samples, out_file.replace(".jsonl", "_base.json"), device0)
    result_json_base["type"] = "base"
    """
    del base_llm
    gc.collect()
    torch.cuda.empty_cache()
    """
    advanced_llm = AutoModelForCausalLM.from_pretrained(
        args.advanced_model, trust_remote_code=True, torch_dtype=torch.float16, cache_dir='/scratch/yl9038/.cache'
    ).to(device1)

    advanced_tokenizer = AutoTokenizer.from_pretrained(
        args.advanced_model,
        use_fast=True,
        trust_remote_code=True,
        cache_dir='/scratch/yl9038/.cache'
    )

    print("Evaluating: advanced")
    result_json_advanced = eval_model(advanced_llm, advanced_tokenizer, samples, out_file.replace(".jsonl", "_advance.json"), device1)
    result_json_advanced["type"] = "advanced"

    print("Evaluating; With classifier")
    result_json_classifier = eval_classifier(base_llm, advanced_llm, base_tokenizer, advanced_tokenizer, samples, out_file.replace(".jsonl", "_classifier.json"), args)
    result_json_classifier["type"] = "With classifier"

    save_jsonl([result_json_base, result_json_advanced, result_json_classifier], out_file)
    # save_jsonl([result_json_base, result_json_advanced], out_file)
    # save_jsonl([result_json_classifier], out_fule.replace(".jsonl","_result_classifier.json"))
    # save_jsonl([result_json_base], out_file.replace(".jsonl","_result_base.json"))
    # save_jsonl([result_json_advance], out_file.replace(".jsonl","_result_advance.json"))

def eval_model(llm, tokenizer, samples, out_file, device):
    all_samples = []
    num_samples = 0
    correct = 0
    start_time = time.time()

    for sample in tqdm(samples, total=len(samples)):
        generated_text = generate_text(llm, tokenizer, sample['prompt'], args, device)
        pred = get_result_from_box(generated_text)
        if pred != None:
            num_samples += 1
            sample['pred'] = pred
            all_samples.append(sample)
            if math_equal(pred, sample['gt']):
                correct += 1

    time_use = time.time() - start_time
    result_json = {
        "num_samples": num_samples,
        "acc": correct / num_samples if num_samples > 0 else 0,
        "time_use_in_second": time_use,
        "time_use_in_minite": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    }

    save_jsonl(all_samples, out_file)
    return result_json

def eval_classifier(base_llm, advanced_llm, base_tokenizer, advanced_tokenizer, samples, out_file, args):
    all_samples = []
    num_samples = 0
    correct = 0
    pred = 0
    start_time = time.time()

    for sample in tqdm(samples, total=len(samples)):
        label = classify(sample['question'])
        if label == 'easy':
            llm, tokenizer, device = base_llm, base_tokenizer, torch.device("cuda:0")
        elif label == 'hard':
            llm, tokenizer, device = advanced_llm, advanced_tokenizer, torch.device("cuda:1")
        elif label == 'very_hard':
            pred = 0
            continue
        else:
            continue
            
        if label in ['easy', 'hard']:
            generated_text = generate_text(llm, tokenizer, sample['prompt'], args, device)
            pred = get_result_from_box(generated_text)

        if pred != None:
            num_samples += 1
            sample['pred'] = pred
            all_samples.append(sample)
            if math_equal(pred, sample['gt']):
                correct += 1

    time_use = time.time() - start_time
    result_json = {
        "num_samples": num_samples,
        "acc": correct / num_samples if num_samples > 0 else 0,
        "time_use_in_second": time_use,
        "time_use_in_minite": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    }

    save_jsonl(all_samples, out_file)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

