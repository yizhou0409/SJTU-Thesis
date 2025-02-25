import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(
        model_name, 
        tokenizer_name=None, 
        device_map="auto", 
        cache_dir="/scratch/yl9038/.cache",
        load_in_half=True,
        padding_side="left",
        use_safetensors=True,
    ):

    if not tokenizer_name:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,padding_side=padding_side, trust_remote_code=True)
    
    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("You are using a new tokenizer without a pad token."
                            "This is not supported by this script.")

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    torch_dtype=torch.float16,
                                                    device_map=device_map,
                                                    trust_remote_code=True,
                                                    use_safetensors=use_safetensors,
                                                    cache_dir=cache_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    if load_in_half:
        model = model.half()
    model.eval()
    return model, tokenizer

def get_layers_to_enumerate(model):
    layers = model.model.layers
    n_layers = len(layers)
    return layers, n_layers