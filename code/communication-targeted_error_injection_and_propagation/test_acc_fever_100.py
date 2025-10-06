from datasets import load_dataset
from datasets import DatasetDict
import random
from run_GUARDIAN_gpt35 import run_llmlp
import re
import time
import json

def load_examples(json_file, num_samples=3, seed=42):
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    random.seed(seed)
    return random.sample(data, num_samples)

ds = load_examples('shared_task_dev.jsonl', num_samples=100, seed=42)

def parse_answer(input_str):
    pattern = r'\b[A-Ca-c]\b'
    matches = re.findall(pattern, input_str)
    return matches[-1].upper() if matches else None

def convert_label_to_option(label):
    label_map = {
        "SUPPORTS": "A",
        "REFUTED": "B", 
        "NOT ENOUGH INFO": "C"
    }
    return label_map.get(label, "Invalid")

def compute_accuracy(gt_label, pred_solution):
    gt_letter = convert_label_to_option(gt_label)
    pred_answer = parse_answer(pred_solution)
    return 1 if gt_letter == pred_answer else 0

cnt = 0
total = 0
total_api_call = 0

for example in ds:
    try:
        question = f"""Please analyze this claim step by step:

Claim: {example['claim']}

1. First, identify the key facts in this claim.
2. Consider if we have sufficient evidence to verify these facts.
3. Make a conclusion based on your analysis.

Choose one option:
A. SUPPORTED
B. REFUTED
C. NOT ENOUGH INFO

Select your answer (A/B/C)."""

        res, api_call = run_llmlp(question)
        total_api_call += api_call
        
        if res is not None:
            cnt += compute_accuracy(example['label'], res)
            total += 1
    except:
        continue

print(f"Accuracy: {cnt / total:.3f}")
print(f"Total API calls: {total_api_call}")