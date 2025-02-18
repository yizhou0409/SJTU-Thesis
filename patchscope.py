import torch
from model import *

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length)
    
    # Decode the output tokens into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response

# Get h_l^i in the execution of source model M on input sequence S
def get_hidden_representation(model, tokenizer, prompt, device) -> torch.Tensor:
    """ Get the residual stream activations of the source model for the input prompt.
    
    Returns:
    hidden_representation: torch.Tensor [batch_size, seq_len, n_layers, hidden_size]
    """
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    layers, n_layers = get_layers_to_enumerate(model)
    hidden_representation = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], n_layers, model.config.hidden_size)).to(device)

    def store_source_activations(layer_id):
        def hook_fn(module, input, output):
            hidden_representation[:, :, layer_id, :] = output[0].detach()
        return hook_fn
    
    hooks = []


    for layer_id, layer in enumerate(layers):
        hook_handle = layer.register_forward_hook(
            store_source_activations(layer_id)
        )
        hooks.append(hook_handle)

    with torch.no_grad():
        _ = source_model(**input_ids)
    for h in hooks:
        h.remove()
    return hidden_representation

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
            print(output[0][:, target_token_position, :], hidden_representation[:, source_token_position, source_layer_id, :])
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
            output_ids = target_model.generate(**inputs, max_length=512)
        
        # Decode the output tokens into text
        generated_text = target_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        hook_handle.remove()
    return generated_text

if __name__=="__main__":
     # Load model and tokenizer
    source_model, source_tokenizer= load_model_and_tokenizer("Qwen/Qwen2.5-Math-1.5B-Instruct")
    target_model, target_tokenizer= load_model_and_tokenizer("Qwen/Qwen2.5-Math-1.5B-Instruct")
    # Ensure CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_prompt = "What is the result of 1*1+1?"
    target_prompt = "What is the result of 1*1+1?"
    print("Response: ", generate_response(source_model, source_tokenizer, source_prompt))
    hidden_representation = get_hidden_representation(source_model, source_tokenizer, source_prompt, device)
    response = patch_target_model(target_model=target_model, target_tokenizer=target_tokenizer, target_prompt=target_prompt, source_layer_id=0, target_layer_id=0, source_token_position=0, target_token_position=0, hidden_representation=f(hidden_representation), device=device)
    print("Modified Response: ", response)