import torch
import torch.nn as nn
from typing import Tuple, Optional # Added Tuple, Optional

class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Expect input as (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0 # Dropout only between GRU layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        # h_prev shape: (num_layers, batch_size, hidden_dim)
        
        if h_prev is None:
            # Initialize hidden state if not provided
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        else:
            h0 = h_prev
        
        # GRU output: (batch_size, seq_len, hidden_dim)
        # hn output: (num_layers, batch_size, hidden_dim)
        gru_out, h_next = self.gru(x, h0)
        
        # Apply the fully connected layer at each timestep (time-distributed)
        # gru_out: (batch_size, seq_len, hidden_dim)
        # prediction_seq: (batch_size, seq_len, output_dim)
        prediction_seq = self.fc(gru_out)
        return prediction_seq, h_next 