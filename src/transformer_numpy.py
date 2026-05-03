import numpy as np

class Transformer_Block:

    def __init__(self, SEQ_LEN, D_MODEL):

        self.SEQ_LEN = SEQ_LEN                                      # Sequence Length
        self.D_MODEL = D_MODEL    
        
        self.W_q = np.random.randn(self.D_MODEL, self.D_MODEL)      # Query Weight Matrix
        self.W_k = np.random.randn(self.D_MODEL, self.D_MODEL)      # Key Weight Matrix
        self.W_v = np.random.randn(self.D_MODEL, self.D_MODEL)      # Value Weight Matrix                            # Model Dimension
        
        self.GAMMA1  = np.ones((self.D_MODEL,))                     # Learnable Scale for 1st LayerNorm
        self.BETA1 = np.zeros((self.D_MODEL,))                      # Learnable Shift for 1st LaterNorm

        self.W1 = np.random.randn(self.D_MODEL, 4*self.D_MODEL)     # Weight Matrix of 1st Layer in FFN
        self.b1 = np.random.randn(4*self.D_MODEL)                   # Bias matrix of 1st Layer in FFN
        self.W2 = np.random.randn(4*self.D_MODEL, self.D_MODEL)     # Weight Matrix of 2nd Layer in FFN
        self.b2 = np.random.randn(self.D_MODEL)                     # Bias matrix of 2nd Layer in FFN

        self.GAMMA2  = np.ones((self.D_MODEL,))                     # Learnable Scale for 2nd LayerNorm
        self.BETA2 = np.zeros((self.D_MODEL,))                      # Learnable Shift for 2nd LaterNorm

    def softmax(self, x):

        x = x - np.max(x, axis=1, keepdims=True)
        sum = np.sum(np.exp(x), axis=1, keepdims=True)

        sftmx = np.exp(x)/sum

        return sftmx
    
    def layer_norm(self, x, gamma, beta):

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x = (x - mean)/np.sqrt(var + 1e-5)

        return ((x * gamma) + beta)
    
    def relu(self, x):
        return np.maximum(x, 0)
    
    def FFN(self, x, w1, b1, w2, b2):
        return ((self.relu((x @ w1) + b1) @ w2) + b2)
    
    def forward(self, x, verbose=False):
        if verbose:
            print("-------------------------------")
            print("--- NumPy Transformer Block ---")
            print("-------------------------------\n")
            print(f"Input Shape = {x.shape}\n")
            print(f"Size of QKV Matrices")
            print(f"W_q = {self.W_q.shape}")
            print(f"W_k = {self.W_k.shape}")
            print(f"W_v = {self.W_v.shape}")
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Calculate Attention
        atten = self.softmax((Q @ K.T)/np.sqrt(self.D_MODEL)) @ V

        if verbose:
            print(f"\nShape of Initial Attention Matrix = {atten.shape}")

        # Add residual
        residual = x
        atten += residual
        if verbose:
            print(" Added Residual")

        # Layer Norm 1
        atten = self.layer_norm(atten, self.GAMMA1, self.BETA1)
        if verbose:
            print(" Added 1st Layer Norm")

        residual = atten

        # Feed Forward Network
        atten = self.FFN(atten, self.W1, self.b1, self.W2, self.b2)
        if verbose:
            print(" Passed thru FFN")

        # Add residual
        atten += residual

        # Layer Norm 2
        atten = self.layer_norm(atten, self.GAMMA2, self.BETA2)
        if verbose:
            print(" Added 2nd Layer Norm")

            print(f"Shape of Final Attention Matrix = {atten.shape}")
        return atten
    

if __name__ == "__main__":

    seq_len = 128
    d_model = 256

    transformer = Transformer_Block(seq_len, d_model)

    inp = np.random.randn(seq_len, d_model)

    transformer.forward(inp, verbose=True)