import torch
from model import *
from utils import *
from tqdm import tqdm
import argparse

# Get h_l^i in the execution of source model M on input sequence S
def get_hidden_representation(source_model, source_tokenizer, prompt, n_tokens, device) -> torch.Tensor:
    """ Get the residual stream activations of the source model for the input prompt.
    
    Returns:
    hidden_representation: torch.Tensor [batch_size, seq_len, n_layers, hidden_size]
    """

    input_ids = source_tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    layers, n_layers = get_layers_to_enumerate(source_model)
    hidden_representation = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], n_layers, source_model.config.hidden_size)).to(device)

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

            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            predicted_text += source_tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            
            input_ids = {"input_ids": torch.cat([input_ids["input_ids"], predicted_token_id.unsqueeze(-1)], dim=-1)}

    for h in hooks:
        h.remove()
    return hidden_representation, predicted_text

# f(h, /theta): Rd -> Rd*
def f(h):
    return h #Temporarily use identity mapping


# Replace h* with f(h) and continue forward pass
def patch_target_model(
                        target_model,
                        target_tokenizer,
                        target_prompt,
                        source_layer_id: int,
                        target_layer_id: int,
                        target_token_position: int,
                        source_token_position: int,
                        hidden_representation: torch.Tensor,
                        device
) -> torch.Tensor:

    def patching_handler(source_layer_id: int,
                            target_token_position: int,
                            source_token_position: int):
        def patching_hook(module, input, output):
            output[0][:, target_token_position, :] = hidden_representation[:, source_token_position, source_layer_id, :]
            return output
        return patching_hook

    layers, n_layers = get_layers_to_enumerate(target_model)
    hook_handle = layers[target_layer_id].register_forward_hook(
        patching_handler(source_layer_id=source_layer_id, 
                            target_token_position=target_token_position,
                            source_token_position=source_token_position)
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
    
    return predicted_token


def patchscope(
        samples, 
        source_model, 
        source_tokenizer, 
        target_model, 
        target_tokenizer, 
        device, 
        source_layer_id, 
        target_layer_id, 
        target_token_position, 
        source_token_position
):
    """
    Operates patchscope on given samples:
    1. Checks if the target model generates the correct answer without patching.
    2. If correct, applies patching and checks the new prediction.
    
    Args:
    - samples: List of sample dictionaries containing prompts and ground truth answers.
    - source_model: The model providing hidden representations.
    - source_tokenizer: Tokenizer for the source model.
    - target_model: The model to be patched.
    - target_tokenizer: Tokenizer for the target model.
    - device: The computation device (CPU/GPU).
    - source_layer_id: The layer index from which to extract the hidden representation.
    - target_layer_id: The layer index in the target model where patching is applied.
    - target_token_position: Token index in the target model where the patching is applied.
    - source_token_position: Token index in the source model where the hidden state is extracted.
    
    Returns:
    - results: A list containing results for each sample.
    """
    results = []
    for sample in tqdm(samples, total=len(samples)):
        idx = sample["idx"]
        source_prompt = sample["source_prompt"]
        target_prompt = sample["target_prompt"]
        ground_truth = sample["gt"]
        n_digits = get_digit(ground_truth)
        # Get hidden representation and unpatched prediction in a single pass
        hidden_representation, unpatched_prediction = get_hidden_representation(
            source_model, source_tokenizer, source_prompt, n_digits, device
        )

        # Step 2: If correct, apply patching and check the new prediction
        patched_prediction = patch_target_model(
            target_model, target_tokenizer, target_prompt,
            source_layer_id, target_layer_id,
            target_token_position, source_token_position,
            hidden_representation, device
        )

        results.append({
            "idx": idx,
            "question": sample["question"],
            "gt": ground_truth,
            "unpatched_pred": unpatched_prediction,
            "patched_pred": patched_prediction
        })

    return results

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
        target_prompt = sample["target_prompt"]
        ground_truth = sample["gt"]
        n_digits = get_digit(ground_truth)
        # Get hidden representation and unpatched prediction in a single pass
        hidden_representation, unpatched_prediction = get_hidden_representation(
            source_model, source_tokenizer, source_prompt, n_digits, device
        )
        hidden_representations[idx] = hidden_representation
        # Step 2: If correct, apply patching and check the new prediction
        patched_prediction = patch_target_model(
            target_model, target_tokenizer, target_prompt,
            source_layer_id, target_layer_id,
            target_token_position, source_token_position,
            hidden_representation, device
        )

        source_token_position = 0
        target_token_position = args.target_token_position
        if args.eval_source_token=="last_digit":
            source_token_position = get_digit_token_position(source_prompt, source_tokenizer)
        elif args.eval_source_token=="last_word":
            source_token_position = get_word_token_position(source_prompt, source_tokenizer)
        sample["source_token_position"] = source_token_position
        sample["target_token_position"] = target_token_position

        results.append({
            "idx": idx,
            "question": sample["question"],
            "gt": ground_truth,
            "unpatched_pred": unpatched_prediction,
            "patched_pred": None,
        })
        if ground_truth == unpatched_prediction:
            remain_samples.append(sample)


    print("Now working on each layers:")
    _, n_layers = get_layers_to_enumerate()

    target_layer_id = args.target_layer_id

    accuracy = []
    length = len(remain_samples)
    for layer_id in range(n_layers):

        print("Layer {}".format(layer_id))
        source_layer_id = layer_id
        correct = 0

        for sample in tqdm(remain_samples, total = length):
               
            source_token_position, target_token_position = sample['source_token_position'], sample['target_token_position']
            hidden_representation = hidden_representations[sample['idx']]

            patched_prediction = patch_target_model(
            target_model, target_tokenizer, target_prompt,
            source_layer_id, target_layer_id,
            target_token_position, source_token_position,
            hidden_representation, device
            )

            if patched_prediction == sample["gt"][0]:
                correct += 1
        accuracy.append(correct/length)


    return results, accuracy

def generate_few_shot_prompt(few_shot_examples, new_question):
    """
    Generates a few-shot prompt given example question-answer pairs and a new question.
    
    Args:
        few_shot_examples (list of dict): List of {"question": str, "answer": str} pairs.
        new_question (str): The new question for which the model should generate an answer.
    
    Returns:
        str: A formatted few-shot prompt.
    """
    pre_prompt = "Just answer next question in the following format: \n"
    formatted_examples = "\n".join(
        [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in few_shot_examples]
    )
    prompt = pre_prompt + f" {formatted_examples}\nQ: {new_question}\nA: "
    return prompt

if __name__=="__main__":
     # Load model and tokenizer
    source_model, source_tokenizer= load_model_and_tokenizer("Qwen/Qwen2.5-Math-1.5B-Instruct")
    target_model, target_tokenizer= load_model_and_tokenizer("Qwen/Qwen2.5-Math-1.5B-Instruct")
    # Ensure CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    few_shot_examples = [{"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "answer": "6"},
                        {"question":"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "answer": "5"},
                        ]

    source_prompt = generate_few_shot_prompt(few_shot_examples, "Jeff\u2019s work is 3 miles away.  He walks there and back each day he works.  How many miles does he walk if he has to work 5 times a week?")
    target_prompt = "What is the result of 1*1+1?"
    print("Response: ", generate_response(source_model, source_tokenizer, source_prompt))
