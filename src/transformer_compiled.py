import torch
from src import transformer_pytorch as t_pt

# NOTE: torch.compile requires MSVC (cl.exe) on Windows to JIT-compile CPU kernels.
# Run with the MSVC environment initialized:
#   cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && python -m src.transformer_compiled'

SEQ_LEN = 128
D_MODEL = 256

default_trnsf = torch.compile(t_pt.Transformer_Block(SEQ_LEN, D_MODEL), fullgraph=False, mode="default")
red_ovhd_trnsf = torch.compile(t_pt.Transformer_Block(SEQ_LEN, D_MODEL), fullgraph=False, mode="reduce-overhead")
autotune_trnsf = torch.compile(t_pt.Transformer_Block(SEQ_LEN, D_MODEL), fullgraph=False, mode="max-autotune")

if __name__ == "__main__":

    inp = torch.randn(SEQ_LEN, D_MODEL)

    def_output = default_trnsf.forward(inp)
    print("Default Compiled Transformer works")

    red_ovhd_output = red_ovhd_trnsf.forward(inp)
    print("Reduced Overhead Compiled Transformer works")

    autotune_output = autotune_trnsf.forward(inp)
    print("Autotune Compiled Transformer works")