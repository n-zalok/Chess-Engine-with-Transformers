from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

tokenizer = {'[CLS]': 0, 'WHITE': 1, 'BLACK': 2, 'KINGSIDE_CASTLE': 3,
             'NO_KINGSIDE_CASTLE': 4, 'QUEENSIDE_CASTLE': 5, 'NO_QUEENSIDE_CASTLE': 6,
             'EMPTY': 7, 'P': 8, 'N': 9, 'B': 10, 'R': 11, 'Q': 12, 'K': 13,
             'p': 14, 'n': 15, 'b': 16, 'r': 17, 'q': 18, 'k': 19}

labels = [chr(i) + str(j) for i in range(ord('a'), ord('h') + 1) for j in range(1, 9)]
label_to_id = {label: idx for idx, label in enumerate(labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

label_to_id['e1g1'] = 64
label_to_id['e8g8'] = 64
label_to_id['e1c1'] = 65
label_to_id['e8c8'] = 65

id_to_label[64] = 'O-O'
id_to_label[65] = 'O-O-O'

class Config():
    def __init__(self):
        self.num_hidden_layers = 4
        self.num_attention_heads = 4
        self.max_position_embeddings = 68
        self.vocab_size = 20
        self.hidden_size = 128
        self.intermediate_size = 512
        self.num_classes = 66
        self.embedd_dropout = 0.1
        self.head_dropout = 0.1
        self.multi_dropout = 0.1
        self.FF_dropout = 0.1
        self.classifier_dropout = 0.2
    
    def to_dict(self):
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "embedd_dropout": self.embedd_dropout,
            "head_dropout": self.head_dropout,
            "multi_dropout": self.multi_dropout,
            "FF_dropout": self.FF_dropout,
            "classifier_dropout": self.classifier_dropout
        }

class Embeddings(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings =nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.embedd_dropout)

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).to(self.device)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
def scaled_dot_product_attention(query, key, value, dropout=None):
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    # Apply dropout to attention weights
    weights = F.dropout(weights, p=dropout, training=True)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.dropout = dropout

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state),
            self.v(hidden_state), self.dropout)
        return attn_outputs
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim, config.head_dropout) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.multi_dropout)
    
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.FF_dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.embeddings = Embeddings(config, device)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class ChessMoveClassifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.encoder = TransformerEncoder(config, device)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.start_head = nn.Linear(config.hidden_size, config.num_classes)
        self.end_head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids):
        # Encoder output: (batch, seq_len, hidden_dim)
        x = self.encoder(input_ids)
        pooled = x[:, 0, :]
        pooled = self.dropout(pooled)  # (batch, hidden_dim)

        start_logits = self.start_head(pooled)  # (batch, num_squares)
        end_logits = self.end_head(pooled)      # (batch, num_squares)
        
        return start_logits, end_logits