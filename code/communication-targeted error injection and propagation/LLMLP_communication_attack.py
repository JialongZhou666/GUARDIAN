import math
import random
from LLM_Neuron_communication_attack import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer, generate_answer
from sacrebleu import sentence_bleu
from prompt_lib import GEN_THRESHOLD
import torch
from torch_geometric.data import Data
from torch.optim import Adam
from model_static import DOMINANTDetector
import model_temporal_gib

ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo", device="cuda:0"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        self.device = device

        self.inactive_nodes = set()
        self.inactive_nodes_graph = set()
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer
        elif qtype == "open-ended":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= GEN_THRESHOLD * 100
            self.ans_parser = lambda x: x
        else:
            raise NotImplementedError("Error qtype")

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question, attack_type):
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        def generate_fully_connected_directed_graph(n):
            edge_index = [[], []]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index[0].append(i)
                        edge_index[1].append(j)

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            adj = torch.zeros((n, n), dtype=torch.float32)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adj[i, j] = 1

            return edge_index, adj

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        
        loop_indices = list(range(self.agents))

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            if node_idx in self.inactive_nodes:
                continue

            if idx == 0 and attack_type == "error":
                self.nodes[node_idx].activate_insert_attack(question)
            else:
                self.nodes[node_idx].activate(question)

            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        edge_index, adj = generate_fully_connected_directed_graph(self.agents)
        static_data = Data(
            text=[self.nodes[idx].get_answer() for idx in range(self.agents) if idx in activated_indices],
            edge_index=edge_index,
            adj=adj
        )
        data_t = [static_data]

        model = DOMINANTDetector(hid_dim=128, num_gnn_layers=2).to(self.device)
        optimizer = Adam(model.parameters(), lr=0.001)
        model.fit(static_data, num_epochs=20, optimizer=optimizer)

        max_score, max_node_idx, scores = model.detect(static_data)
        print(f'First Round: Max anomaly score: {max_score}, Node with max score: {max_node_idx}')
        last_max_node_idx = max_node_idx
        
        for rid in range(self.rounds):
            self.inactive_nodes.add(max_node_idx + rid * self.agents)
            self.inactive_nodes_graph.add(max_node_idx)
        
        loop_indices = list(range(self.agents, self.agents*2))

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            if node_idx in self.inactive_nodes:
                continue
            
            if node_idx == 0 or node_idx == 1:
                self.nodes[node_idx].activate_insert_attack(question)
            else:
                self.nodes[node_idx].activate(question)

            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        edge_index, adj = generate_fully_connected_directed_graph(self.agents - len(self.inactive_nodes_graph))
        current_data = Data(
            text=[self.nodes[idx].get_answer() for idx in range(self.agents, self.agents*2) if idx in activated_indices],
            edge_index=edge_index,
            adj=adj
        )
        for previous_data in data_t:
            previous_data.text.pop(last_max_node_idx)
            previous_data.edge_index = edge_index
            previous_data.adj = adj
        data_t.append(current_data)

        model = model_temporal_gib.train_model(data_t)
        max_score, max_node_idx, scores = model.detect(data_t)
        print(f'Second Round: Max anomaly score: {max_score}, Node with max score: {max_node_idx}')
        last_max_node_idx = max_node_idx
        temp = 0
        for node in self.inactive_nodes_graph:
            if node <= max_node_idx:
                temp += 1
        max_node_idx += temp
        for rid in range(self.rounds):
            self.inactive_nodes.add(max_node_idx + rid * self.agents)
            self.inactive_nodes_graph.add(max_node_idx)
        
        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if node_idx in self.inactive_nodes:
                    continue
                    
                if idx in idx_mask:
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

            edge_index, adj = generate_fully_connected_directed_graph(self.agents - len(self.inactive_nodes_graph))
            current_data = Data(
                text=[self.nodes[idx].get_answer() for idx in range(self.agents * rid, self.agents * (rid + 1)) if idx in idxs],
                edge_index=edge_index,
                adj=adj
            )
            for previous_data in data_t:
                previous_data.text.pop(last_max_node_idx)
                previous_data.edge_index = edge_index
                previous_data.adj = adj
            data_t.append(current_data)

            model = model_temporal_gib.train_model(data_t)
            max_score, max_node_idx, scores = model.detect(data_t)
            print(f'{rid + 1} Round: Max anomaly score: {max_score}, Node with max score: {max_node_idx}')
            last_max_node_idx = max_node_idx
            temp = 0
            for node in self.inactive_nodes_graph:
                if node <= max_node_idx:
                    temp += 1
            max_node_idx += temp
            for rid2 in range(self.rounds):
                self.inactive_nodes.add(max_node_idx + rid2 * self.agents)
                self.inactive_nodes_graph.add(max_node_idx)

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens