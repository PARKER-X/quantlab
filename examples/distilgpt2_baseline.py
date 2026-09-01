import time
from pathlib import Path
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

PROMPT = (
    "Artificial intelligence is changing the way "
    "people work and learn."
    "What is llm and how does it work?"
)

Max_new_tokens = 50
seed = 42

res_dir = Path("experiments/results")
res_files = res_dir/"distilgpt2_fp32_baseline.json"

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("DistilGPT2 FP32 Baseline")
print("="*70)
print("\nDevice:")
print(device)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

parameter_count = sum(p.numel() for p in model.parameters())
trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_dtype = str(next(model.parameters()).dtype)

print("\nModel:")
print(model_name)

print("\nParameters:")
print(f"{parameter_count:,}")

print("\nTrainable Parameters:")
print(f"{trainable_parameter_count:,}")

print("\nModel Dtype:")
print(model_dtype)

parameter_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
parameter_mb = parameter_bytes / (1024 ** 2)

print("\nParameter Memory:")
print(f"{parameter_mb:.2f} MB")

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

print("\nPrompt:")
print(PROMPT)

print("\nInput IDs:")
print(inputs["input_ids"])

print("\nInput Token Count:")
print(inputs["input_ids"].shape[1])


print("\nRunning warm-up...")
with torch.no_grad():
    _ = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\nRunning baseline inference...")

if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=Max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.perf_counter()
latency_seconds = end_time - start_time

generated_token_count = outputs.shape[1] - inputs["input_ids"].shape[1]

tokens_per_second = ( generated_token_count / latency_seconds if latency_seconds > 0 else 0)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:")
print("-" * 70)
print(generated_text)
print("-" * 70)

print("\nBenchmark:")
print(f"Latency: {latency_seconds:.4f} seconds")
print(f"Generated Tokens: {generated_token_count}")
print(f"Tokens/Second: {tokens_per_second:.2f}")

res_dir.mkdir(parents=True, exist_ok=True)

results = {
    "experiment": "distilgpt2_fp32_baseline",
    "model": model_name,
    "quantization": "none",
    "precision": "FP32",
    "device": str(device),
    "seed": seed,
    "prompt": PROMPT,
    "max_new_tokens": Max_new_tokens,
    "parameter_count": parameter_count,
    "trainable_parameter_count": trainable_parameter_count,
    "model_dtype": model_dtype,
    "parameter_memory_mb": round(parameter_mb, 4),
    "input_token_count": int(
        inputs["input_ids"].shape[1]
    ),
    "generated_token_count": int(
        generated_token_count
    ),
    "latency_seconds": round(
        latency_seconds,
        6
    ),
    "tokens_per_second": round(
        tokens_per_second,
        4
    ),
    "generated_text": generated_text,
}

with open(res_files, "w") as f:
    import json
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(res_files)

print("\n" + "=" * 70)
print("Baseline experiment complete")
print("=" * 70)