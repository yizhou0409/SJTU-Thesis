import torch
from model import *
from utils import *
from tqdm import tqdm
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
import re


#  h_l^i in the execution of source model M on input sequence S
def get_hidden_representation(source_model, source_tokenizer, prompt, n_tokens, device) -> torch.Tensor:
    """ Get the residual stream activations of the source model for the input prompt.
    
    Returns:
    hidden_representation: torch.Tensor [batch_size, seq_len, n_layers, hidden_size]
    """
    input_ids = source_tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    layers, n_layers = get_layers_to_enumerate(source_model)
    hidden_representation = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], n_layers, source_model.config.hidden_size))

    def store_source_activations(layer_id):
        def hook_fn(module, input, output):
            hidden_representation[:, :, layer_id, :] = output[0].detach()
        return hook_fn
    
    hooks = [layer.register_forward_hook(store_source_activations(layer_id)) for layer_id, layer in enumerate(layers)]

    predicted_text = ""
    with torch.no_grad():
        for _ in range(n_tokens):
            outputs = source_model(**input_ids)
            logits = outputs.logits  # (batch_size, seq_length, vocab_size)

            if _ == 0:  # Store hidden representation only for the first token
                hidden_representation = hidden_representation.clone()
                for h in hooks:
                    h.remove()
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            predicted_text += source_tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            
            input_ids = {"input_ids": torch.cat([input_ids["input_ids"], predicted_token_id.unsqueeze(-1)], dim=-1)}

    for h in hooks:
        h.remove()
    return hidden_representation, predicted_text

def get_hidden_representation_cot_eval(source_model, source_tokenizer, prompt, device):
        # Prepare input
    input_ids = source_tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    layers, n_layers = get_layers_to_enumerate(source_model)
    hidden_representation = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], n_layers, source_model.config.hidden_size))
    
    def store_source_activations(layer_id):
        def hook_fn(module, input, output):
            hidden_representation[:, :, layer_id, :] = output[0].detach()
        return hook_fn
    
    hooks = [layer.register_forward_hook(store_source_activations(layer_id)) for layer_id, layer in enumerate(layers)]
    first_token_hidden = []
    pre_result_hidden = []
    predicted_text = ""
    flag = 0
    with torch.no_grad():
        for _ in range(512):
            outputs = source_model(**input_ids)
            logits = outputs.logits
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            predicted_text += source_tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            input_ids = {"input_ids": torch.cat([input_ids["input_ids"], predicted_token_id.unsqueeze(-1)], dim=-1)}
            if _ == 0:
                first_token_hidden = hidden_representation.clone()
                for h in hooks:
                    h.remove()
            if "\\boxed{" in predicted_text and flag == 0:
                hidden_representation = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], n_layers, source_model.config.hidden_size))
                hooks = [layer.register_forward_hook(store_source_activations(layer_id)) for layer_id, layer in enumerate(layers)]
                flag += 1
            if "\\boxed{" in predicted_text and flag == 1:
                pre_result_hidden = hidden_representation.clone()
                for h in hooks:
                    h.remove()
                flag += 1
            if re.search(r"\\boxed{.*?}", predicted_text):
                break

    return first_token_hidden, pre_result_hidden, predicted_text

# f(h, /theta): Rd -> Rd*
def f(h):
    return h #Temporarily use identity mapping


# Replace h* with f(h) and continue forward pass
def patch_target_model(
                        target_model,
                        target_tokenizer,
                        target_prompt,
                        target_layer_id: int,
                        target_token_id: int,
                        hidden_representation: torch.Tensor,
                        device
) -> torch.Tensor:

    def patching_handler(source_layer_id: int,
                            target_token_id: int,
                            source_token_id: int):
        def patching_hook(module, input, output):
            output[0][:, target_token_id, :] = f(hidden_representation)
            return output
        return patching_hook

    layers, n_layers = get_layers_to_enumerate(target_model)
    hook_handle = layers[target_layer_id].register_forward_hook(
        patching_handler(source_layer_id=source_layer_id, 
                            target_token_id=target_token_id,
                            source_token_id=source_token_id)
    ) 

    try:
        inputs = target_tokenizer(target_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = target_model(**inputs)  # Get logits instead of generating tokens

        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        last_token_logits = logits[:, -1, :]  # Extract logits for the last token

        predicted_token_id = torch.argmax(last_token_logits, dim=-1)  # Get the most probable next token
        predicted_token = target_tokenizer.decode(predicted_token_id, skip_special_tokens=True)

    finally:
        hook_handle.remove()
    
    return predicted_token, last_token_logits

def patchscope_eval(
        samples, 
        source_model, 
        source_tokenizer, 
        target_model, 
        target_tokenizer, 
        device, 
        args
):
    hidden_representations = {}
    results = []
    remain_samples = []
    print("Getting the hidden representation......")

    for sample in tqdm(samples, total=len(samples)):
        idx = sample["idx"]
        source_prompt = sample["source_prompt"]
        ground_truth = sample["gt"]
        n_digits = get_digit(ground_truth)
        # Get hidden representation and unpatched prediction in a single pass
        hidden_representation, unpatched_prediction = get_hidden_representation(
            source_model, source_tokenizer, source_prompt, n_digits, device
        )
        hidden_representations[idx] = hidden_representation.cpu()

        source_token_id = 0
        target_token_id = args.target_token_id
        if args.eval_source_token=="last_digit":
            source_token_id = last_digit_token_id(source_prompt, source_tokenizer)
        elif args.eval_source_token=="last_word":
            source_token_id = last_word_token_id(source_prompt, source_tokenizer)
        elif args.eval_source_token=="last":
            source_token_id = -1
        elif args.eval_source_token=="use_arg":
            source_token_id = args.source_token_id
        sample["source_token_id"] = source_token_id
        sample["target_token_id"] = target_token_id

        if ground_truth == unpatched_prediction:
            remain_samples.append(sample)


    print("Now working on each layers:")
    _, n_layers = get_layers_to_enumerate(source_model)

    target_layer_id = args.target_layer_id

    accuracy = []
    surprisal = []
    length = len(remain_samples)
    for layer_id in range(n_layers):
        if args.eval_target_layer == 'same':
            target_layer_id = layer_id
        print("Layer {}".format(layer_id), "Target_layer {}".format(target_layer_id))
        correct = 0
        surprise = 0
        for sample in tqdm(remain_samples, total = length):
               
            hidden_representation = hidden_representations[sample['idx']].to(device)
            patched_prediction, last_token_logits = patch_target_model(
            target_model, target_tokenizer, sample["target_prompt"],
            target_layer_id,
            sample["target_token_id"], 
            hidden_representation[:, sample["source_token_id"], layer_id, :], device
            )

            patched_surprisal = compute_surprisal(last_token_logits, target_tokenizer.encode(sample["gt"][0]))
            surprise += patched_surprisal

            if patched_prediction[0] == sample["gt"][0]:
                correct += 1
            if layer_id == n_layers - 1:
                result = {
                    "idx": sample["idx"],
                    "question": sample["question"],
                    "gt": sample["gt"],
                    "patched_pred": patched_prediction,
                    "target_prompt": sample["target_prompt"],
                    "source_id" : sample["source_token_id"],
                    "surprisal" : patched_surprisal
                }
                results.append(result)                
        print("Accuracy: ", correct/length, "Surprisal: ", surprise/length)
        surprisal.append(surprise/length)
        accuracy.append(correct/length)


    return results, accuracy, surprisal

def patchscope_eval_cot(samples, source_model, source_tokenizer, target_model, target_tokenizer, device, args):
    """
    Evaluates a chain-of-thought (CoT) prompt, capturing hidden representations
    at the first generated token and just before the result token (\boxed{}).
    """
    hidden_representations = {}
    results = []
    remain_samples = []
    print("Getting the hidden representation......")
    for sample in tqdm(samples, total=len(samples)):
        idx = sample["idx"]
        source_prompt = sample["source_prompt"]
        ground_truth = sample["gt"]
        # Get hidden representation and unpatched prediction in a single pass
        first_token_hidden, pre_result_hidden, output_text = get_hidden_representation_cot_eval(source_model, source_tokenizer, source_prompt, device)
        hidden_representations[idx] = [first_token_hidden, pre_result_hidden]
        sample["first_token"] = output_text.strip(" ")[0]
        sample["target_token_id"] = args.target_token_id
        if ground_truth == get_result_from_box(output_text):
            remain_samples.append(sample)

    print("Now working on each layers:")
    _, n_layers = get_layers_to_enumerate(source_model)

    target_layer_id = args.target_layer_id

    accuracy_first = []
    accuracy_r = []
    surprisal_first = []
    surprisal_r = []
    length = len(remain_samples)

    for layer_id in range(n_layers):
        if args.eval_target_layer == 'same':
            target_layer_id = layer_id
        print("Layer {}".format(layer_id), "Target_layer {}".format(target_layer_id))
        correct_first = 0
        surprise_first = 0
        correct_r = 0
        surprise_r = 0

        for sample in tqdm(remain_samples, total = length):
            first_token_hidden, pre_result_hidden = hidden_representations[sample['idx']][0], hidden_representations[sample['idx']][1]
            # Check first token
            patched_prediction_first, first_logits = patch_target_model(
            target_model, target_tokenizer, sample["target_prompt"],
            target_layer_id,
            sample["target_token_id"], 
            pre_result_hidden[:, -1, layer_id, :], device
            )

            if patched_prediction_first == sample["first_token"]:
                correct_first += 1
            surprise_first += compute_surprisal(first_logits, target_tokenizer.encode(sample["first_token"]))

            # Check result
            patched_prediction_result, result_logits = patch_target_model(
            target_model, target_tokenizer, sample["target_prompt"],
            target_layer_id,
            args.target_token_id,
            pre_result_hidden[:, -1, layer_id, :], device
            )

            if patched_prediction_result[0] == sample["gt"][0]:
                correct_r += 1
            surprise_r += compute_surprisal(result_logits, target_tokenizer.encode(sample["gt"][0]))
            if layer_id == n_layers - 1:
                result = {
                    "idx": sample["idx"],
                    "question": sample["question"],
                    "gt": sample["gt"],
                    "patched_pred": patched_prediction_result,
                    "target_prompt": sample["target_prompt"],
                    "source_id" : sample["source_token_id"],
                    "surprisal" : compute_surprisal(result_logits, target_tokenizer.encode(sample["gt"][0])),
                    "first_token_gt" : sample["first_token"],
                    "first_token_predict" : patched_prediction_first,
                }
                results.append(result)                
        print("Accuracy: ", correct/length, "Surprisal: ", surprise/length)
        surprisal_first.append(surprise_first/length)
        surprisal_r.append(surprise_r/length)
        accuracy_first.append(correct_first/length)
        accuracy_r.append(correct_r/length)
    
    return results, accuracy_first, accuracy_r, surprisal_first, surprisal_r


