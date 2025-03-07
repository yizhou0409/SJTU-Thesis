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
    parser.add_argument("--data_names", default="gsm8k", type=str) #gsm8k, math
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--source_model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--target_model_name", default="same", type=str) # same or specify a model name
    parser.add_argument("--eval_target_layer", default="same", type=str) #use_arg, same
    parser.add_argument("--eval_first_token", action="store_false")
    parser.add_argument("--eval_numbers", action="store_false")
    parser.add_argument("--eval_operators", action="store_false")
    parser.add_argument("--eval_batchsize", default=128, type=int) # Use batchsize to prevent OOM
    parser.add_argument("--target_layer_id", default=-1, type=int)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str) #qwen25-math-cot, direct
    parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_token_gen", default=512, type=int)
    parser.add_argument("--shuffle", action='store_true')

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
        eval(source_model, target_model, source_tokenizer, target_tokenizer, data_name, args, device)

def eval(source_model, target_model, source_tokenizer, target_tokenizer, data_name, args, device):

    _, n_layers = get_layers_to_enumerate(source_model)

    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # Prepare samples, prompts, target prompts
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

    batchsize = args.eval_batchsize
    
    results = []
    num_samples, num_operators, num_numbers = 0, 0, 0
    correct_result, surprisal_result = [0 for i in range(n_layers)], [0 for i in range(n_layers)],
    correct_first, surprisal_first = [0 for i in range(n_layers)] if args.eval_first_token else [], [0 for i in range(n_layers)] if args.eval_first_token else []
    correct_operators, surprisal_operators = [0 for i in range(n_layers)] if args.eval_operators else [], [0 for i in range(n_layers)] if args.eval_operators else []
    correct_numbers, surprisal_numbers = [0 for i in range(n_layers)] if args.eval_numbers else [], [0 for i in range(n_layers)] if args.eval_numbers else []

    # Accumulate corrects and surprisal from samples
    for i in range(0, len(samples), batchsize):
        print("Batch {} of {}".format(i//batchsize, len(samples)//batchsize))
        samples_batch = samples[i: i+batchsize]
        num_samples_batch, results_batch, correct_result_batch, surprisal_result_batch, correct_first_batch, surprisal_first_batch, correct_operators_batch, surprisal_operators_batch, correct_numbers_batch, surprisal_numbers_batch, num_operators_batch, num_numbers_batch = patchscope_eval(
                                                                samples_batch, 
                                                                source_model, 
                                                                source_tokenizer, 
                                                                target_model, 
                                                                target_tokenizer, 
                                                                device, 
                                                                args)
        results.extend(results_batch)
        num_samples += num_samples_batch
        num_operators += num_operators_batch
        num_numbers += num_numbers_batch
        correct_result = [a + b for a, b in zip(correct_result, correct_result_batch)]
        correct_first = [a + b for a, b in zip(correct_first, correct_first_batch)]
        correct_operators = [a + b for a, b in zip(correct_operators, correct_operators_batch)]
        correct_numbers = [a + b for a, b in zip(correct_numbers, correct_numbers_batch)]
        surprisal_result = [a + b for a, b in zip(surprisal_result, surprisal_result_batch)]
        surprisal_first = [a + b for a, b in zip(surprisal_first, surprisal_first_batch)]
        surprisal_operators = [a + b for a, b in zip(surprisal_operators, surprisal_operators_batch)]
        surprisal_numbers = [a + b for a, b in zip(surprisal_numbers, surprisal_numbers_batch)]

    # Calculate Result
    accuracy_result = [correct / num_samples for correct in correct_result]
    surprisal_result = [surprise / num_samples for surprise in surprisal_result]
    if args.eval_first_token:
        accuracy_first = [correct / num_samples for correct in correct_first]
        surprisal_first = [surprise / num_samples for surprise in surprisal_first]
    if args.eval_operators:
        accuracy_operators = [correct / num_operators for correct in correct_operators]
        surprisal_operators = [surprise / num_operators for surprise in surprisal_operators]
    if args.eval_numbers:
        accuracy_numbers = [correct / num_numbers for correct in correct_numbers]
        surprisal_numbers = [surprise / num_numbers for surprise in surprisal_numbers]

    accuracy_file_dir = out_file.replace(".jsonl", f"_accuracy_curve.png")
    surprisal_file_dir = out_file.replace(".jsonl", f"_surprisal_curve.png")

    # Print Curves
    accuracy_dict = {
        "Result": accuracy_result,
        "First Token": accuracy_first,
        "Operators": accuracy_operators,
        "Numbers": accuracy_numbers
    }
    plot_accuracy_curves(accuracy_dict, accuracy_file_dir)

    surprisal_dict = {
        "Result": surprisal_result,
        "First Token": surprisal_first,
        "Operators": surprisal_operators,
        "Numbers": surprisal_numbers
    }
    plot_surprisal_curves(surprisal_dict, surprisal_file_dir)
    
    # Save result Json                              
    with open(out_file, "w") as file:
        json.dump(results, file, indent=4)

    return

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
