from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import math
from accelerate import Accelerator
from itertools import islice
from tqdm import tqdm
import SelfExtend

accelerator = Accelerator()
device = accelerator.device

model_name = 'meta-llama/Llama-2-7b-chat-hf'
auth_token = "[your HF token]"  # replace with your token if needed

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_auth_token=auth_token
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)

window_size = 512
group_size = 16
use_flash = True

# Apply SelfExtend
SelfExtend.apply(model, group_size, window_size, rate, enable_flash_attention=use_flash, flash_attention_impl="flash_attn")
model.eval()

# Load PG19 test set
dataset = load_dataset("pg19", split="test", streaming=True)

# Helper function to compute perplexity
def calculate_perplexity(text, max_length=4096, stride=512):
    encodings = tokenizer(text, return_tensors="pt")
    
    print("Encoded")

    input_ids = encodings.input_ids
    
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_slice = input_ids[:, begin_loc:end_loc]
        target_slice = input_slice.clone()
        target_slice[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_slice, labels=target_slice)
            neg_log_likelihood = outputs.loss * stride

        print(neg_log_likelihood)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(1))
    return ppl.item()

context_sizes = [4096, 6144, 8192, 10240, 12288, 14336, 16384]

example = next(iter(dataset))  # This gives you the first book

results = []

for size in context_sizes:
    perplexities = []

    text = example["text"]
    try:
        ppl = calculate_perplexity(text, max_length = size)
        perplexities.append(ppl)
    except Exception as e:
        print("Error:", e)

# Final result
    mean_ppl = sum(perplexities) / len(perplexities)
    results.append(mean_ppl)
    print(f"Context length {size}: {mean_ppl:.2f}")

print(results)

