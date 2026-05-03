import numpy as np

SEQ_LEN = 128
D_MODEL = 256

print(f"SEQUENCE LENGTH -> SEQ_LEN = {SEQ_LEN}")
print(f"MODEL DIMENSION -> D_MODEL = {D_MODEL}")

# Input -> Sequence of tokens
# Each token is represented as a vector of size "d_model"
# So Input is a 2D array of shape (seq_len, d_model)
# For now, we create a random input sequence.

# Initialize a 2D array with random numbers
inp = np.random.randn(SEQ_LEN, D_MODEL)
print(f"INPUT SIZE = {inp.shape}")

# Attention requires Q, K, V Matrices.
# Each one projects the input from D_MODEL -> D_MODEL
# If input is (128, 256), the size of QKV matrices will have to be (256, 256)

W_q = np.random.randn(D_MODEL, D_MODEL)
W_k = np.random.randn(D_MODEL, D_MODEL)
W_v = np.random.randn(D_MODEL, D_MODEL)

print(f"SIZE OF QKV WEIGHT MATRICES")
print(f"W_q = {W_q.shape}")
print(f"W_k = {W_k.shape}")
print(f"W_v = {W_v.shape}")

# Atten = softmax(QK.T / √d_k) . V

Q = inp @ W_q   # (128, 256) * (256, 256) -> (128, 256)
K = inp @ W_k   # K : (128, 256)
V = inp @ W_v   # V : (128, 256)

def softmax(x):

    # The size of x here will be (128, 128)
    # Q*KT -> (128, 256)*(256, 128) -> (128, 128)
    # Each row represents one token's attn scores wrt to all other positions
    # So sum should be across rows 

    x = x - np.max(x, axis=1, keepdims=True)
    sum = np.sum(np.exp(x), axis=1, keepdims=True)

    # axis=1 means sum is row-wise
    # NumPy will collapse that axis and give you a 1D array (128,)
    # This is a problem since we also need to divide each value for softmax
    # If we divide directly, NumPy will broadcast along columns not rows
    # By using keepdims=True, the sum stays (128, 1)
    # Now each row will get divided by its sum

    # Extra: Numerical Stability fix

    sftmx = np.exp(x)/sum

    return sftmx

# Calculate Attention
atten = softmax((Q @ K.T)/np.sqrt(D_MODEL)) @ V

print(f"Shape of Initial Attention Matrix = {atten.shape}")

# Residual Connection
def add_residual_conn(x, inp):   
    return x + inp

atten = add_residual_conn(atten, inp)
print("--- Added 1st Residual Connection ---")

# Layer Normalization

GAMMA1  = np.ones((D_MODEL,))
BETA1 = np.zeros((D_MODEL,))
# These values change during training, but here they will remain the same

def layer_norm(x, GAMMA, BETA):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    # We need to compute the mean and var across the feature dimension
    # axis=-1 chooses the last dimension which in our case (128, 256) is the 256 features
    # This way we will get the mean and var in features for all tokens

    x = (x - mean)/np.sqrt(var + 1e-5)

    # After normalization, every token's features have mean=0 and std=1
    # That can be very rigid. The network could need a different scale or offset
    # for a particular layer to work well. A learnable scale (gamma) and a shift (beta)
    # give it that flexibility

    return ((x * GAMMA) + BETA)

atten = layer_norm(atten, GAMMA1, BETA1)
print("--- Added 1st Layer Normalization ---")

# Feed Forward Neural Network

# This is basically two linear layers with a ReLU in between:
# FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

# inp is in shape (128, 256)
# W1 projects from d_model -> 4*d_model i.e. 256 -> 1024
# Therefore, W1 will have to be (256, 1024)

W1 = np.random.randn(D_MODEL, 4*D_MODEL)

# b1 will have shape of (1024,)

b1 = np.random.randn(4*D_MODEL)

# W2 projects from 4 * d_model -> d_model i.e. 1024 -> 256
# Therefore, W2 will have to be (1024, 256)

W2 = np.random.randn(4*D_MODEL, D_MODEL)

# b2 will have shape of (256,)

b2 = np.random.randn(D_MODEL)

def relu(x):
    return np.maximum(x, 0)

def FFN(x):
    return ((relu((x @ W1) + b1) @ W2) + b2)

residual = atten.copy()

atten = FFN(atten)
print("--- Added 1st FFN ---")

# Residual Connection

atten = add_residual_conn(atten, residual)
print("--- Added 2nd Residual Connection ---")

# Another Layer Norm
GAMMA2  = np.ones((D_MODEL,))
BETA2 = np.zeros((D_MODEL,))
atten = layer_norm(atten, GAMMA2, BETA2)
print("--- Added 2nd Layer Normalization ---")

print(f"Shape of Attention Matrix = {atten.shape}")