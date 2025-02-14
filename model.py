import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        cache_dir="$SCRATCH/.cache"
        load_in_half=True,
        padding_side="left",
        use_safetensors=True,
    ):

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, padding_side=padding_side, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, legacy=False, use_fast=use_fast_tokenizer, padding_side=padding_side, trust_remote_code=True)

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

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
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