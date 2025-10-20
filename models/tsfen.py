import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8, temperature=0.08):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output_projection(context)

        return output, attention_weights


class LSTMTemporalExtractor(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x)
        lstm_output = self.layer_norm(lstm_output)
        return lstm_output


class TSFEN(nn.Module):
    def __init__(self, d_model=64, num_heads=8, hidden_size=128, num_agents=20):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_agents = num_agents

        self.positional_encoding = PositionalEncoding(d_model)
        self.input_projection = nn.Linear(3, d_model)

        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, temperature=0.08)
        self.lstm_extractor = LSTMTemporalExtractor(input_size=d_model, hidden_size=hidden_size, num_layers=2)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_agents)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def forward(self, state_tensor):
        batch_size, seq_len, num_agents, feature_dim = state_tensor.shape

        state_tensor = state_tensor.view(batch_size * seq_len * num_agents, feature_dim)
        projected = self.input_projection(state_tensor)
        projected = projected.view(batch_size, seq_len * num_agents, self.d_model)

        with_pos = self.positional_encoding(projected)

        attention_output, attention_weights = self.mhsa(with_pos)

        lstm_output = self.lstm_extractor(attention_output)

        final_output = lstm_output[:, -1, :]

        actor_logits = self.actor_head(final_output)
        critic_value = self.critic_head(final_output)

        return actor_logits, critic_value, attention_weights

    def apply_adaptive_flmd_mask(self, flmd_values, beta=1.0, mu_over_L=0.01, current_round=1):
        device = next(self.parameters()).device
        flmd_tensor = torch.tensor(flmd_values, dtype=torch.float32, device=device)
        
        adaptive_mask = torch.exp(-beta * flmd_tensor * ((1 - mu_over_L) ** (-current_round)))
        return adaptive_mask

