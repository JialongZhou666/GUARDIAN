from datasets import load_dataset
from datasets import DatasetDict
import random
from run_GUARDIAN_gpt35 import run_llmlp
import re
import time
import json
from utils import is_equiv, extract_math_answer

ds = load_dataset("lighteval/MATH", "all")

all_test_examples = []
for example in ds['test']:
    all_test_examples.append(example)

random.seed(42)
sampled_examples = random.sample(all_test_examples, 100)

def compute_accuracy(gt_solution, pred_solution):
    try:
        gt_answer = extract_math_answer(gt_solution)
        if not gt_answer:
            return 0
            
        pred_answer = extract_math_answer(pred_solution)
        if not pred_answer:
            return 0
        
        return 1 if is_equiv(gt_answer, pred_answer) else 0
        
    except:
        return 0

cnt = 0
total = 0
total_api_call = 0

for example in sampled_examples:
    try:
        prompt = f"Question: {example['problem']}\nPlease solve this step by step and give the final answer as 'The answer is [number]'."
        
        res, api_call = run_llmlp(prompt)
        total_api_call += api_call
        
        if res is not None:
            cnt += compute_accuracy(example['solution'], res)
            total += 1
            
    except:
        continue

print(f"Accuracy: {cnt/total:.3f}")
print(f"Total API calls: {total_api_call}")