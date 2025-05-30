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
        
        # We only want the output from the last time step for prediction
        # If x is a single time step (seq_len=1), out[:, -1, :] correctly takes that step's output.
        # out shape: (batch_size, hidden_dim)
        out_last_step = gru_out[:, -1, :] 
        
        # Pass through the fully connected layer
        # out shape: (batch_size, output_dim)
        prediction = self.fc(out_last_step)
        return prediction, h_next 