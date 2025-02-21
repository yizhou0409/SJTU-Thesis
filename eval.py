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
from model import load_model_and_tokenizer
from patchscope import *
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--source_model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--source_layer_id", default=0, type=int)
    parser.add_argument("--target_layer_id", default=0, type=int)
    parser.add_argument("--source_token_id", default=-1, type=int)
    parser.add_argument("--target_token_id", default=-1, type=int)    
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="direct", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--use_safetensors", action="store_false")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


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
    model_name = "/".join(args.model_name.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

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


def setup(args):
    source_model, source_tokenizer = load_model_and_tokenizer(
        model_name=args.source_model_name,
        load_in_half=True,
        use_fast_tokenizer=True,
        use_safetensors=args.use_safetensors,
    )
    target_model, target_tokenizer = load_model_and_tokenizer(
        model_name=args.target_model_name,
        load_in_half=True,
        use_fast_tokenizer=True,
        use_safetensors=args.use_safetensors,
    )    

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(source_model=source_model,
                            target_model=target_model,
                            source_tokenizer=source_tokenizer,
                            target_tokenizer=target_tokenizer,
                            data_name=data_name,
                            args=args,))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["accuracy_patched"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['accuracy_patched']:.1f}".ljust(pad, " ") for result in results]))


def main(source_model, source_tokenizer, target_model, target_tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    executer=PythonExecutor(get_answer_from_student=True)

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
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": gt_ans,
            "source_prompt": source_full_prompt,
            "target_prompt": target_full_prompt
        }

        samples.append(sample)

    samples_of_samples = random.sample(samples, args.n_samples)

    outputs = patchscope(samples_of_samples, 
                        source_model, 
                        source_tokenizer, 
                        target_model, 
                        target_tokenizer, 
                        device, 
                        source_layer_id, 
                        target_layer_id, 
                        target_token_position, 
                        source_token_position)

    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=4)

    # Process all outputs
    result_json = evaluate(outputs)
    time.use = time.time - start_time
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )
    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)

    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
