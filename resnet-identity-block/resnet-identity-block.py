import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        x_permuted = np.moveaxis(x, 1, -1)
        z1 = x_permuted @ self.W1.T
        a1 = relu(z1)
        
        z2 = a1 @ self.W2.T
        y_permuted = relu(z2) + x_permuted
        y = np.moveaxis(y_permuted, -1, 1)
        return y
