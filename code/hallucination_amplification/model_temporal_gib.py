import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, output_dim)
        
    def forward(self, texts):
        if isinstance(texts, (int, float)):
            texts = [str(texts)]
        elif isinstance(texts, (list, tuple)):
            texts = [str(t) for t in texts]
        elif not isinstance(texts, str):
            raise ValueError(f"Expected number, string or list, got {type(texts)}")
            
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(next(self.bert.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.bert.parameters()).device)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        return self.linear(pooled_output)

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        deg = torch.zeros(num_nodes, dtype=x.dtype, device=device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=x.dtype, device=device))
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt[deg_inv_sqrt != deg_inv_sqrt] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = torch.zeros_like(x)
        src = x[col] * norm.view(-1, 1)
        out.index_add_(0, row, src)
        
        return self.linear(out)

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        h = self.conv2(x, edge_index)
        
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

class GNNDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GNNDecoder, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.linear1(z))
        return self.linear2(h)

class StructureDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(StructureDecoder, self).__init__()
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, z):
        num_nodes = z.size(0)
        device = z.device
        
        rows = []
        cols = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                rows.append(i)
                cols.append(j)
        
        rows = torch.tensor(rows, device=device)
        cols = torch.tensor(cols, device=device)
        
        node_pairs = torch.cat([z[rows], z[cols]], dim=1)
        
        h = F.relu(self.linear1(node_pairs))
        probs = torch.sigmoid(self.linear2(h))
        
        adjacency = probs.view(num_nodes, num_nodes)
        return adjacency

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        return self.transformer(x)

class TemporalDOMINANT(nn.Module):
    def __init__(self, hid_dim, num_gnn_layers, num_transformer_layers, nhead):
        super(TemporalDOMINANT, self).__init__()
        self.text_encoder = TextEncoder(hid_dim)
        self.gnn_encoder = GNNEncoder(hid_dim, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, nhead, num_transformer_layers)
        self.gnn_decoder = GNNDecoder(hid_dim, hid_dim)
        self.structure_decoder = StructureDecoder(hid_dim)
        
    def forward(self, data_list):
        device = next(self.parameters()).device
        
        node_features = []
        mus = []
        logvars = []
        
        for data in data_list:
            text_features = self.text_encoder(data.text)
            features, mu, logvar = self.gnn_encoder(text_features, data.edge_index.to(device))
            node_features.append(features)
            mus.append(mu)
            logvars.append(logvar)
        
        sequence = torch.stack(node_features)
        temporal_features = self.transformer(sequence)
        
        last_features = temporal_features[-1]
        x_reconstructed = self.gnn_decoder(last_features)
        adjacency_reconstructed = self.structure_decoder(last_features)
        
        last_mu = mus[-1]
        last_logvar = logvars[-1]
        
        return x_reconstructed, adjacency_reconstructed, last_mu, last_logvar

class TemporalDOMINANTDetector(nn.Module):
    def __init__(self, hid_dim, num_gnn_layers, num_transformer_layers, nhead, 
                 feature_weight=0.5, beta=0.001):
        super(TemporalDOMINANTDetector, self).__init__()
        self.model = TemporalDOMINANT(hid_dim, num_gnn_layers, num_transformer_layers, nhead)
        self.feature_weight = feature_weight
        self.structure_weight = 1 - feature_weight
        self.beta = beta
        
    def forward(self, data_list):
        x_reconstructed, adjacency_reconstructed, mu, logvar = self.model(data_list)
        original_x = self.model.text_encoder(data_list[-1].text)
        original_adjacency = data_list[-1].adj

        feature_scores = torch.mean((x_reconstructed - original_x) ** 2, dim=1)
        structure_scores = torch.mean((adjacency_reconstructed - original_adjacency) ** 2, dim=1)
        scores = self.feature_weight * feature_scores + self.structure_weight * structure_scores

        gib_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = torch.mean(scores) + self.beta * gib_loss

        return total_loss, scores

    def fit(self, data_list, num_epochs, optimizer):
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss, _ = self.forward(data_list)
            loss.backward()
            optimizer.step()

    def detect(self, data_list):
        self.model.eval()
        with torch.no_grad():
            _, scores = self.forward(data_list)
            max_score = torch.max(scores)
            max_node_idx = torch.argmax(scores)
        return max_score.item(), max_node_idx.item(), scores.numpy()

def train_model(train_data_list):
    model = TemporalDOMINANTDetector(
        hid_dim=64,
        num_gnn_layers=2,
        num_transformer_layers=2,
        nhead=4,
        feature_weight=0.3,
        beta=0.001
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.fit(train_data_list, num_epochs=20, optimizer=optimizer)
    
    return model
