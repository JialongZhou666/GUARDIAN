import ast
import json
import os
import openai
import random
import sys
from prettytable import PrettyTable
from LLMLP_communication_attack import LLMLP
# from LLMLP_static_communication_attack import LLMLP
from utils import *

openai.api_key = ""
openai.api_base = ""

ATTACK_TYPE = "none"
EXP_NAME = "trial_1"
MODEL = "gpt-3.5-turbo"

ACTIVATION = "listwise"
TYPE = "math_exp"
DIR_NAME = "trial"

ROLES = ["Assistant", "Assistant", "Assistant", "Assistant"]

def set_rd_seed(seed):
    random.seed(seed)

def run_llmlp(QUERY):
    set_rd_seed(0)
    assert len(ROLES) > 0

    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL)

    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(QUERY, ATTACK_TYPE)
    
    pt = PrettyTable()
    pt.add_column("Round", ROLES)
    for rid in range(3):
        responses = [(completions[idx][rid] if completions[idx][rid] is not None else "No response.") for idx in range(len(ROLES))]
        pt.add_column(str(rid+1), responses, "l")

    print(r"Query: {}".format(QUERY))
    print(r"#API calls: {}".format(resp_cnt))
    print(r"Prompt Tokens: {}".format(prompt_tokens))
    print(r"Completion Tokens: {}".format(completion_tokens))
    print(pt)
    print(r"Final Answer: {}".format(res))
    print()

    return res, resp_cnt