import torch
from model import *
from utils import *
from tqdm import tqdm
import argparse
import re

def get_hidden_representation(source_model, source_tokenizer, source_prompt, tokens, args, device):
    input_ids = source_tokenizer(source_prompt, return_tensors='pt', truncation=True).to(device, non_blocking=True)
    
    predicted_text = ""
    result_hidden = None

    token_representations = []
    first_token_hidden = None
    first_token_id = None
    flag = True

    with torch.no_grad():
        for _ in range(args.max_token_gen):
            outputs = source_model(**input_ids, output_hidden_states = True)
            logits = outputs.logits
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            predicted_token = source_tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            input_ids = {"input_ids": torch.cat([input_ids["input_ids"], predicted_token_id.unsqueeze(-1)], dim=-1)}

            # Patch Result
            if "\\boxed{" in predicted_text and flag:
                result_hidden = [layer.half().cpu() for layer in outputs.hidden_states]
                flag = False

            # Patch First
            if args.eval_first_token and _ == 0:
                first_token_hidden = (predicted_token_id, [layer.half().cpu() for layer in outputs.hidden_states])

            # Patch Important Tokens
            if predicted_token in tokens:
                if predicted_token == '*':
                    if predicted_text[-1].isdigit() or predicted_text[-2].isdigit():
                        token_representations.append((predicted_token_id, predicted_token, [layer.half().cpu() for layer in outputs.hidden_states]))
                else:
                    token_representations.append((predicted_token_id, predicted_token, [layer.half().cpu() for layer in outputs.hidden_states]))
            predicted_text += predicted_token
            if re.search(r"\\boxed{.*?}", predicted_text):
                break

    return predicted_text, result_hidden, first_token_hidden, token_representations


def f(h):
    return h  # Identity mapping for now

def patch_target_model(target_model, target_tokenizer, target_prompt, target_layer_id: int, hidden_representation: torch.Tensor, device) -> torch.Tensor:
    def patching_handler():
        def patching_hook(module, input, output):
            output[0][:, -1, :] = f(hidden_representation)
            return output
        return patching_hook

    layers, n_layers = get_layers_to_enumerate(target_model)
    hook_handle = layers[target_layer_id].register_forward_hook(patching_handler())
    
    try:
        inputs = target_tokenizer(target_prompt, return_tensors="pt").to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = target_model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        predicted_token_id = torch.argmax(last_token_logits, dim=-1)
    finally:
        hook_handle.remove()
    
    return predicted_token_id, last_token_logits


def patchscope_eval(samples, source_model, source_tokenizer, target_model, target_tokenizer, device, args):

    # Build Important Token List
    operators = ["+","-","*","/"]
    numbers = [str(i) for i in range(10)]
    tokens = []

    if args.eval_numbers:
        tokens += numbers
    if args.eval_operators:
        tokens += operators

    hidden_representations = {}
    results = []
    remain_samples = []
    print("Getting the hidden representation......")
    
    batch_size = 16  # Tune based on GPU memory
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch_samples = samples[i:i + batch_size]
        for sample in batch_samples:
            idx = sample["idx"]
            source_prompt = sample["source_prompt"]
            ground_truth = sample["gt"]
            predicted_text, result_hidden, first_token_hidden, token_representations = get_hidden_representation(
                source_model, source_tokenizer, source_prompt, tokens, args, device)
            sample["predicted"] = get_result_from_box(predicted_text)
            if args.eval_wrong_answer:
                if get_result_from_box(predicted_text) and not math_equal(get_result_from_box(predicted_text), ground_truth):
                    hidden_representations[idx] = (result_hidden, first_token_hidden, token_representations)
                    remain_samples.append(sample)
            else:
                if math_equal(get_result_from_box(predicted_text), ground_truth):
                    hidden_representations[idx] = (result_hidden, first_token_hidden, token_representations)
                    remain_samples.append(sample)
    
    print("Now working on each layer:")
    _, n_layers = get_layers_to_enumerate(source_model)
    target_layer_id = args.target_layer_id
    
    corrects_first, corrects_operators, corrects_numbers, corrects_result, surprisal_first, surprisal_operators, surprisal_numbers, surprisal_result = [], [], [], [], [], [], [], []
    num_samples = len(remain_samples)
    

    for layer_id in range(n_layers):
        if args.eval_target_layer == 'same':
            target_layer_id = layer_id
        print("Layer {}".format(layer_id), "Target_layer {}".format(target_layer_id))
        correct_first, surprise_first, correct_result, surprise_result = 0, 0, 0, 0
        correct_operators, correct_numbers, surprise_operators, surprise_numbers = 0, 0, 0, 0
        num_operators, num_numbers = 0, 0
        for sample in tqdm(remain_samples, total=num_samples):
            patched_prediction, logits = None, None

            # Eval result
            pre_result_hidden = hidden_representations[sample['idx']][0][layer_id][:, -1, :].to(device, non_blocking=True)
            patched_prediction, logits = patch_target_model(target_model, target_tokenizer, sample["target_prompt"], target_layer_id, pre_result_hidden, device)
            if list(patched_prediction) == target_tokenizer.encode(sample["predicted"][0]):
                correct_result += 1
            surprise_result += compute_surprisal(logits, target_tokenizer.encode(sample["predicted"][0]))
            
            # Eval first token
            if args.eval_first_token:
                first_token_id, first_token_hidden = hidden_representations[sample['idx']][1][0], hidden_representations[sample['idx']][1][1][layer_id][:, -1, :].to(device, non_blocking=True)
                patched_prediction, logits = patch_target_model(target_model, target_tokenizer, sample["target_prompt"], target_layer_id, first_token_hidden, device)
                if patched_prediction == first_token_id:
                    correct_first += 1
                surprise_first += compute_surprisal(logits, first_token_id) 

            # Eval important tokens
            token_representations = hidden_representations[sample['idx']][2]
            for token_id, token, representation in token_representations:
                representation = representation[layer_id][:, -1, :].to(device, non_blocking=True)
                patched_prediction, logits = patch_target_model(target_model, target_tokenizer, sample["target_prompt"], target_layer_id, representation, device)
                if token in operators:
                    num_operators += 1
                    if patched_prediction == token_id:
                        correct_operators += 1
                    surprise_operators += compute_surprisal(logits, token_id)
                elif token in numbers:
                    num_numbers += 1
                    if patched_prediction == token_id:
                        correct_numbers += 1
                    surprise_numbers += compute_surprisal(logits, token_id)
               
            if layer_id == n_layers - 1:
                result = {
                    "idx": sample["idx"],
                    "question": sample["question"],
                    "gt": sample["gt"],
                    "target_prompt": sample["target_prompt"],
                    "predicted": sample["predicted"],
                }
                results.append(result)

        corrects_result.append(correct_result)
        surprisal_result.append(surprise_result)

        if args.eval_first_token:
            corrects_first.append(correct_first)
            surprisal_first.append(surprise_first)
        
        if args.eval_operators:
            corrects_operators.append(correct_operators)
            surprisal_operators.append(surprise_operators)
        
        if args.eval_numbers:
            corrects_numbers.append(correct_numbers)
            surprisal_numbers.append(surprise_numbers)
    
    return num_samples, results, corrects_result, surprisal_result, corrects_first, surprisal_first, corrects_operators, surprisal_operators, corrects_numbers, surprisal_numbers, num_operators, num_numbers
