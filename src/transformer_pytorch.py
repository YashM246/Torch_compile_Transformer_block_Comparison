import torch
import torch.nn as nn

class Transformer_Block(nn.Module):

    def __init__(self, SEQ_LEN, D_MODEL):
        super().__init__()

        self.SEQ_LEN = SEQ_LEN
        self.D_MODEL = D_MODEL

        # We initialized raw weight matrices in NumPy (W_q, W_k, W_v)
        # Using PyTorch's nn.Linear we can convert the raw matrices to Linear
        # transformations. 
        # nn.Linear applies y = xA.T + b to the incoming data
        # For our case b will be 0 
        # This also makes it torch.compile compatible and helps in Gradient tracking later

        self.W_q = nn.Linear(self.D_MODEL, self.D_MODEL, bias=False)
        self.W_k = nn.Linear(self.D_MODEL, self.D_MODEL, bias=False)
        self.W_v = nn.Linear(self.D_MODEL, self.D_MODEL, bias=False)

        # Similarly, we initialize two LayerNorm Layers
        
        self.LayerNorm1 = nn.LayerNorm(self.D_MODEL)
        self.LayerNorm2 = nn.LayerNorm(self.D_MODEL)

        # And Initialize two Linear Layers for the FFN

        self.Linear1 = nn.Linear(self.D_MODEL, 4*self.D_MODEL)
        self.Linear2 = nn.Linear(4*self.D_MODEL, self.D_MODEL)

    def forward(self, x, verbose=False):
        if verbose:
            print("-------------------------------")
            print("--- Torch Transformer Block ---")
            print("-------------------------------\n")
            print(f"Input Shape = {x.shape}\n")
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        residual = x

        # Calculate Attention

        # atten = torch.softmax((Q @ K.T)/torch.sqrt(torch.tensor(self.D_MODEL, dtype=torch.float32)), dim=-1) @ V
        # This will only handle (seq_len, d_model) sized 2D inputs
        # In case of batching the shape will be (batch_size, seq_len, d_model)
        # Therefore .T will not work
        # Using .T will create a shape of (d_model, seq_len, batch_size)

        atten = torch.softmax((Q @ torch.transpose(K, -1, -2))/torch.sqrt(torch.tensor(self.D_MODEL, dtype=torch.float32)), dim=-1) @ V

        # Now the transpose will create a shape of (batch_size, d_model, seq_len)
        # This will work for 2D as well as 3D inputs

        if verbose:
            print(f"\nShape of Initial Attention Matrix = {atten.shape}")

        # Add residual

        atten += residual
        if verbose:
            print(" Added Residual")
        
        # Layer Norm 1

        atten = self.LayerNorm1(atten)
        if verbose:
            print(" Added 1st Layer Norm")

        residual = atten

        # FFN

        atten = self.Linear1(atten)
        atten = torch.relu(atten)
        atten = self.Linear2(atten)
        if verbose:
            print(" Passed thru FFN")

        # Add residual

        atten += residual

        # Layer Norm 2

        atten = self.LayerNorm2(atten)
        if verbose:
            print(" Added 2nd Layer Norm")

            print(f"Shape of Final Attention Matrix = {atten.shape}")
        
        return atten
    
if __name__ == "__main__":

    seq_len = 128
    d_model = 256

    transformer = Transformer_Block(seq_len, d_model)

    inp = torch.randn(seq_len, d_model)

    transformer.forward(inp, verbose=True)