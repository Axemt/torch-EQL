from typing import List
import torch
from torch import nn
from torch.nn import functional as F

# From https://github.com/martius-lab/EQL/blob/331f6023b195e5b93ba7ec191763a116f241d4b7/EQL-DIV-ICML-Python3/src/mlfg_final.py#L139

class BaseFuncLayer(nn.Module):
    """
    A `torch.nn` Module providing `basefunc[i](W*z1+b, [W*z2+b])`

    This layer applies available functions to all incoming features after applying a weight (and possibly bias).
    For functions with greater arity than 1, all combinations are tested. Currently only 2-ary functions are supported

    Supported functions and their indeces:
        1-ary:
            0: `id(z)`
            1: `sin(z)`
            2: `cos(z)`
        2-ary:
            0: `id(z1, z2)` [= z1]
            1: `z1 * z2`

    Args:
        n_in: The number of input features
        n_per_base: The number of `basefunc`s per input feature

    Kwargs:
        bias: Whether or not to use bias
        basefuncs_1ary: The indices of the associated base 1-ary functions to enable. By default all enabled
        basefuncs_2ary: The indices of the associated base 2-ary functions to enable. By default all enabled

    """

    def __init__(
            self,
            in_features: int,
            n_per_base: int,
            bias: bool = True,
            basefuncs_1ary: List[int] = None,
            basefuncs_2ary: List[int] = None
        ) -> None:
        super().__init__()

        if basefuncs_1ary is None:
            basefuncs_1ary = [0, 1, 2]
        if basefuncs_2ary is None:
            basefuncs_2ary = [0, 1]

        self.n_base_1ary = len(basefuncs_1ary)
        self.n_base_2ary = len(basefuncs_2ary)
        self.n_per_base = n_per_base
        self.n_in = in_features

        self.out_features = (self.n_base_1ary + self.n_base_2ary) * n_per_base
        
        n_w = (self.n_base_1ary + 2 * self.n_base_2ary) * n_per_base

        self.W = nn.Parameter(
            torch.randn(size=(in_features, n_w)),
            requires_grad=True
        )

        self.b = nn.Parameter(
            torch.zeros(size=(n_w,)),
            requires_grad=bias
        )

        self.fun_1ary_idx = nn.Parameter( torch.Tensor( basefuncs_1ary ).repeat(n_per_base) , requires_grad=False)
        self.fun_2ary_idx = nn.Parameter( torch.Tensor( basefuncs_2ary ).repeat(n_per_base) , requires_grad=False)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # 1. Apply wX+b
        node_input = torch.matmul(input, self.W ) + self.b


        # 2. Prepare tensors to apply basefuncs to
        z  = node_input[
            :, 
            :self.n_per_base * self.n_base_1ary
        ]
        
        z1 = node_input[
            :, 
            self.n_per_base*self.n_base_1ary : self.n_per_base * (self.n_base_1ary + self.n_base_2ary)
        ]
        z2 = node_input[
            :,
            self.n_per_base * (self.n_base_1ary + self.n_base_2ary) :
        ]

        # 3. Apply basefuncs
        fn_1ary = torch.where(
            # idx == 0 -> identity
            torch.eq(self.fun_1ary_idx, 0), z,
            torch.where(
                # idx == 1 -> sin
                torch.eq(self.fun_1ary_idx, 1), torch.sin(z),
                # else -> cos
                torch.cos(z),
            )
        )

        fn_2ary = torch.where(
            # idx == 0 -> id
            torch.eq(self.fun_2ary_idx, 0), z1,
            # else -> a * b
            z1 * z2,
        )

        # 4. Reshape into single column tensor output with f1ary, f2ary results
        out = torch.concat([fn_1ary, fn_2ary], dim=1)

        return out
    
class EQL(nn.Module):

    def __init__(
            self,
            in_features: int,
            n_fglayers: int,
            n_per_base: int,
            bias: bool = True,
            basefuncs_1ary: List[int] | None = None,
            basefuncs_2ary: List[int] | None = None,
        ) -> None:
        super().__init__()

        last_in_dim = in_features
        fglayers = []
        for _ in range(n_fglayers):
            layer = BaseFuncLayer(
                    in_features=last_in_dim,
                    n_per_base=n_per_base,
                    bias=bias,
                    basefuncs_1ary=basefuncs_1ary,
                    basefuncs_2ary=basefuncs_2ary
            )
            fglayers.append(layer)
            last_in_dim = layer.out_features

        self.n_fglayers = n_fglayers
        self.n_per_base = n_per_base
        self.hidden_layers = nn.Sequential(
            *fglayers,
        )
        self.linear =  nn.Linear(fglayers[-1].out_features, 1, bias=bias)

        # *2? see https://github.com/samuelkim314/DeepSymRegTorch/blob/fe17a19cf339e2dee9870f6fb8045eec90849f32/utils/symbolic_network.py#L226
        #self.out_W = nn.Parameter( torch.rand( size=(fglayers[-1].out_features, 1) ) * 2, requires_grad=True )
        #self.out_b = nn.Parameter( torch.ones(size=(1,)) , requires_grad=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        out = self.hidden_layers(input)        
        out = self.linear(out)


        return out
        