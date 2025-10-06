from datasets import load_dataset
from datasets import DatasetDict
import random
from run_GUARDIAN_gpt35 import run_llmlp
import re
import time
import json
import traceback

ds = load_dataset("cais/mmlu", "all")

all_test_examples = []
for example in ds['test']:
    all_test_examples.append(example)

random.seed(42)
sampled_examples = random.sample(all_test_examples, 100)

def parse_answer(input_str):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)
    
    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        pattern_no_parentheses = r'\b[A-Da-d]\b'
        matches_no_parentheses = re.findall(pattern_no_parentheses, input_str)
        
        if matches_no_parentheses:
            solution = matches_no_parentheses[-1].upper()

    return solution

def compute_accuracy(gt, pred_solution):
    gt_map = ['A', 'B', 'C', 'D']
    
    try:
        gt_letter = gt_map[gt]
    except IndexError:
        return 0

    pred_answer = parse_answer(pred_solution)
    
    return 1 if gt_letter == pred_answer else 0

cnt = 0
total = 0
total_api_call = 0

for example in sampled_examples:
    try:
        question_with_choices = example['question'] + " The answer options range from: " + ", ".join(example['choices']) + ". Corresponding to the four options (A), (B), (C), and (D) respectively."
        res, api_call = run_llmlp(question_with_choices)
        total_api_call += api_call
        
        if res is not None:
            cnt += compute_accuracy(example['answer'], res)
            total += 1
            
    except:
        continue

print(f"Accuracy: {cnt/total:.3f}")
print(f"Total API calls: {total_api_call}")