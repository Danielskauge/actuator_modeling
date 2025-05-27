import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers.
    
    Features:
    - Configurable activation function
    - Dropout for regularization
    - Batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Initialize MLP with configurable architecture.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Define activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create layers
        self.layers = nn.ModuleList()
        
        print(f"input_dim: {input_dim}", type(input_dim))
        print(f"hidden_dims[0]: {hidden_dims[0]}", type(hidden_dims[0]))
        
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Batch norm for input layer if requested
        if use_batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dims[0])])
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass through the MLP."""

        # Iterate through layers and corresponding batch norm layers (if applicable)
        if self.use_batch_norm:
            # Ensure bn_layers has the same length as layers
            assert len(self.layers) == len(self.bn_layers), "Mismatch between number of layers and batch norm layers"
            
            for i, (layer, bn) in enumerate(zip(self.layers, self.bn_layers)):

                x = layer(x)
                x = bn(x) # Apply batch norm directly
                x = self.activation(x)            
                x = self.dropout(x)
        else:
            # Original loop structure if not using batch norm
            for i, layer in enumerate(self.layers):

                x = layer(x)
                # No batch norm here
                x = self.activation(x)

                x = self.dropout(x)

        # Output layer (no activation or batch norm)
        x = self.output_layer(x)

        return x 