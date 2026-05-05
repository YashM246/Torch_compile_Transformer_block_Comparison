import numpy as np
import torch
from src import transformer_numpy as t_np
from src import transformer_pytorch as t_pt

# The goal of this script is to ensure that forward passes in both Numpy and 
# PyTorch version of our transformer give the same output given the same weights
# and input

# Set seq_len and d_model

SEQ_LEN = 128
D_MODEL = 256

# Initialize weights and parameters

q_weights = np.random.randn(D_MODEL, D_MODEL)
k_weights = np.random.randn(D_MODEL, D_MODEL)
v_weights = np.random.randn(D_MODEL, D_MODEL)

# Learnable Scale and Shift are 1 and 0 respectively in both so we dont need
# to change them here

w1_weights = np.random.randn(D_MODEL, 4*D_MODEL)
b1_values = np.random.randn(4*D_MODEL)

w2_weights = np.random.randn(4*D_MODEL, D_MODEL)
b2_values = np.random.randn(D_MODEL)

# Initialize Transformer Blocks

Numpy_Transformer = t_np.Transformer_Block(SEQ_LEN=SEQ_LEN, D_MODEL=D_MODEL)
Pytorch_Transformer = t_pt.Transformer_Block(SEQ_LEN=SEQ_LEN, D_MODEL=D_MODEL).double()

# Allign weights and parameters for Numpy Implementation

Numpy_Transformer.W_q = q_weights
Numpy_Transformer.W_k = k_weights
Numpy_Transformer.W_v = v_weights
Numpy_Transformer.W1 = w1_weights
Numpy_Transformer.b1 = b1_values
Numpy_Transformer.W2 = w2_weights
Numpy_Transformer.b2 = b2_values

# Allign weights and parameters for PyTorch Implementation

# Weight matrices are implemented as nn.Linear layers and 
Pytorch_Transformer.W_q.weight.data = torch.from_numpy(q_weights.T)
Pytorch_Transformer.W_k.weight.data = torch.from_numpy(k_weights.T)
Pytorch_Transformer.W_v.weight.data = torch.from_numpy(v_weights.T)
Pytorch_Transformer.Linear1.weight.data = torch.from_numpy(w1_weights.T)
Pytorch_Transformer.Linear1.bias.data = torch.from_numpy(b1_values)
Pytorch_Transformer.Linear2.weight.data = torch.from_numpy(w2_weights.T)
Pytorch_Transformer.Linear2.bias.data = torch.from_numpy(b2_values)

# Initialize Input

inp = np.random.randn(SEQ_LEN, D_MODEL)

#  Forward Pass

np_output = Numpy_Transformer.forward(inp, verbose= False)
pt_output = Pytorch_Transformer.forward(torch.from_numpy(inp), verbose= False)

# Check if values match
print((pt_output - torch.from_numpy(np_output)).abs().max().item())

if torch.allclose(pt_output, torch.from_numpy(np_output), atol=1e-10):
    print("--- Values Match ---")
    print(f"\nLargest difference between two: {(pt_output - torch.from_numpy(np_output)).abs().max().item()}\n")
else:
    print("--- Values Do not Match ---")
    print(f"\nLargest difference between two: {(pt_output - torch.from_numpy(np_output)).abs().max().item()}\n")