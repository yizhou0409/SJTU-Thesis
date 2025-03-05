import random
import os
import argparse
import torch
import json
import gc

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import *
from parser import *
from data_loader import *
from model import load_model_and_tokenizer
from patchscope import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--source_model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--target_model_name", default="same", type=str) # same or specify a model name
    parser.add_argument("--eval_target_layer", default="same", type=str) #use_arg, same
    parser.add_argument("--target_layer_id", default=-1, type=int)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str) #qwen25-math-cot, direct
    parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--num_test_sample", default=20, type=int)  # -1 for full data
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--n_samples", default=500, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    return args


def setup(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_model, source_tokenizer = load_model_and_tokenizer(
        model_name=args.source_model_name,
        load_in_half=True,
    )
    target_model, target_tokenizer = source_model, source_tokenizer #If same model, save CUDA memory
    if args.target_model_name != 'same':
        target_model, target_tokenizer = load_model_and_tokenizer(
            model_name=args.target_model_name,
            load_in_half=True,
        )    

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        source_model = torch.nn.DataParallel(source_model)
        target_model = torch.nn.DataParallel(target_model)

    data_list = args.data_names.split(",")

    print("Mode: Eval, Prompt_type: {}".format(args.prompt_type))
    for data_name in data_list:
        torch.cuda.empty_cache()  # Clear memory before processing a new dataset
        gc.collect()  # Force garbage collection
        main_eval(source_model, target_model, source_tokenizer, target_tokenizer, data_name, args, device)

def main_eval(source_model, target_model, source_tokenizer, target_tokenizer, data_name, args, device):
    examples, out_file = prepare_data(data_name, args)
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
        example["gt_ans"] = gt_ans

        source_full_prompt = construct_prompt(example, data_name, args)
        target_full_prompt = generate_target_prompt(target_tokenizer)

        if idx == args.start:
            print(source_full_prompt)

        if get_digit(gt_ans):
            sample = {
                "idx": idx,
                "question": example["question"],
                "gt": gt_ans,
                "source_prompt": source_full_prompt,
                "target_prompt": target_full_prompt
            }
            samples.append(sample)

    if 'cot' in args.prompt_type:
        results, accuracy_first, accuracy_r, surprisal_first, surprisal_r = patchscope_eval_cot(samples, 
                                                                                                source_model, 
                                                                                                source_tokenizer, 
                                                                                                target_model, 
                                                                                                target_tokenizer, 
                                                                                                device, 
                                                                                                args)
        accuracy_r_file_dir = out_file.replace(".jsonl", f"_accuracy_r_curve.png")
        accuracy_first_file_dir = out_file.replace(".jsonl", f"_accuracy_first_curve.png")
        surprisal_r_file_dir = out_file.replace(".jsonl", f"_surprisal_r_curve.png")
        surprisal_first_file_dir = out_file.replace(".jsonl", f"_surprisal_first_curve.png")
        plot_accuracy_curve(accuracy_r, accuracy_r_file_dir)
        plot_accuracy_curve(accuracy_first, accuracy_first_file_dir)
        plot_surprisal_curve(surprisal_r, surprisal_r_file_dir)        
        plot_surprisal_curve(surprisal_first, surprisal_first_file_dir)                                                                   
    else:
        results, accuracy, surprisal = patchscope_eval(samples, 
                                                    source_model, 
                                                    source_tokenizer, 
                                                    target_model, 
                                                    target_tokenizer, 
                                                    device, 
                                                    args)

        accuracy_file_dir = out_file.replace(".jsonl", f"_accuracy_curve.png")
        surprisal_file_dir = out_file.replace(".jsonl", f"_surprisal_curve.png")
        plot_accuracy_curve(accuracy, accuracy_file_dir)
        plot_surprisal_curve(surprisal, surprisal_file_dir)

    with open(out_file, "w") as file:
        json.dump(results, file, indent=4)

    return

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
