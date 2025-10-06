import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, text_list):
        if not text_list:
            return torch.zeros((0, self.fc.out_features), device=self.bert.device)
        
        if isinstance(text_list, torch.Tensor):
            text_list = text_list.tolist()
            
        processed_texts = []
        for text in text_list:
            if text is None:
                processed_texts.append("")
            elif isinstance(text, (int, float)):
                processed_texts.append(str(text))
            elif isinstance(text, list):
                processed_texts.append(" ".join(str(t) for t in text if t is not None))
            elif isinstance(text, str):
                processed_texts.append(text)
            else:
                processed_texts.append(str(text))
        
        processed_texts = [text if text.strip() else " " for text in processed_texts]
        
        try:
            encoded_input = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            encoded_input = {k: v.to(self.bert.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = self.bert(**encoded_input)
            
            text_features = outputs.last_hidden_state[:, 0, :]
            return self.fc(text_features)
            
        except Exception as e:
            print("Error in TextEncoder forward pass:")
            print("Input text_list:", text_list)
            print("Processed texts:", processed_texts)
            print("Error message:", str(e))
            raise

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super(GNNEncoder, self).__init__()
        self.conv_layers = nn.ModuleList([GCNConv(input_dim if i == 0 else hid_dim, hid_dim) for i in range(num_layers)])

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

class DOMINANT(nn.Module):
    def __init__(self, hid_dim, num_gnn_layers):
        super(DOMINANT, self).__init__()
        self.text_encoder = TextEncoder(hid_dim)
        self.gnn_encoder = GNNEncoder(hid_dim, hid_dim, num_gnn_layers)
        self.attribute_decoder = nn.Linear(hid_dim, hid_dim)
        self.structure_decoder = nn.Linear(hid_dim, hid_dim)

    def forward(self, data):
        x = self.text_encoder(data.text)

        z = self.gnn_encoder(x, data.edge_index)
        
        x_reconstructed = self.attribute_decoder(z)
        
        adjacency_reconstructed = torch.sigmoid(torch.matmul(self.structure_decoder(z), self.structure_decoder(z).t()))
        
        return x_reconstructed, adjacency_reconstructed

class DOMINANTDetector(nn.Module):
    def __init__(self, hid_dim, num_gnn_layers, feature_weight=0.3):
        super(DOMINANTDetector, self).__init__()
        self.model = DOMINANT(hid_dim, num_gnn_layers)
        self.feature_weight = feature_weight
        self.structure_weight = 1 - feature_weight

    def forward(self, data):
        device = next(self.parameters()).device
        data.edge_index = data.edge_index.to(device)
        data.adj = data.adj.to(device)
        
        x_reconstructed, adjacency_reconstructed = self.model(data)
        
        original_x = self.model.text_encoder(data.text)
        original_adjacency = data.adj

        feature_loss = F.mse_loss(x_reconstructed, original_x)
        structure_loss = F.binary_cross_entropy(adjacency_reconstructed, original_adjacency)

        loss = self.feature_weight * feature_loss + self.structure_weight * structure_loss

        feature_scores = torch.mean((x_reconstructed - original_x) ** 2, dim=1)
        structure_scores = torch.mean((adjacency_reconstructed - original_adjacency) ** 2, dim=1)
        scores = self.feature_weight * feature_scores + self.structure_weight * structure_scores

        return loss, scores

    def fit(self, data, num_epochs, optimizer):
        self.model.train()
        device = next(self.parameters()).device
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss, _ = self.forward(data.to(device))
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    def detect(self, data):
        self.model.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            _, scores = self.forward(data.to(device))
        
        scores = scores.cpu()
        max_score, max_node = torch.max(scores, dim=0)
        return max_score.item(), max_node.item(), scores.numpy()
        