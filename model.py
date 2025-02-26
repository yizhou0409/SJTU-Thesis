import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def last_digit_token_position(sentence: str, tokenizer) -> int:
    tokens = tokenizer.tokenize(sentence)
    token_spans = tokenizer(sentence, return_offsets_mapping=True)["offset_mapping"]

    # Find all digits and their character positions in the original sentence
    digit_positions = [m.start() for m in re.finditer(r'\d', sentence)]
    if not digit_positions:
        return None

    last_digit_pos = digit_positions[-1]  # Position of the last digit in the original sentence

    # Find the corresponding token index
    for i, (start, end) in enumerate(token_spans):
        if start <= last_digit_pos < end:
            return -(len(tokens) - i)  # Negative indexing

    return None

def last_word_token_position(sentence: str, tokenizer) -> int:
    tokens = tokenizer.tokenize(sentence)
    token_spans = tokenizer(sentence, return_offsets_mapping=True)["offset_mapping"]

    # Find all words (alphanumeric sequences) and their positions
    word_matches = list(re.finditer(r'\b\w+\b', sentence))
    if not word_matches:
        return None

    last_word_pos = word_matches[-1].start()  # Position of the last word in the original sentence

    # Find the corresponding token index
    for i, (start, end) in enumerate(token_spans):
        if start <= last_word_pos < end:
            return -(len(tokens) - i)  # Negative indexing

    return None

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_text

def predict_next_token(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, seq_length, vocab_size)

    last_token_logits = logits[:, -1, :]  # Get last token logits

    predicted_token_id = torch.argmax(last_token_logits, dim=-1)  # Most probable token
    predicted_text = tokenizer.decode(predicted_token_id, skip_special_tokens=True)

    return predicted_text


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