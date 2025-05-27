import torch
import torch.nn as nn

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        # h0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU output: (batch_size, seq_len, hidden_dim)
        # hn output: (num_layers, batch_size, hidden_dim)
        out, _ = self.gru(x, h0)
        
        # We only want the output from the last time step for prediction
        # out shape: (batch_size, hidden_dim)
        out = out[:, -1, :] 
        
        # Pass through the fully connected layer
        # out shape: (batch_size, output_dim)
        out = self.fc(out)
        return out 