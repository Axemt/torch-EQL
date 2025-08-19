from typing import List
import torch
from torch import nn
import numpy as np
import sympy as sp
from warnings import warn

# See https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function

class RelaxedDivision(torch.autograd.Function):
    # FIXME: RelaxedDiv vs symbolic div fails verification by a very small error?

    threshold = 1e-7

    @staticmethod
    def forward(ctx, z1: torch.Tensor, z2: torch.Tensor):
        print('hi')
        # 1. Create a mask for all values that are in the range (-threshold, threshold)
        mask = (torch.abs(z2) < RelaxedDivision.threshold).type(torch.float32)
        # 2. If they are within the threshold range, apply an 'eps'-like value moving it 
        #     away from the singularity point at 0, in the same direction
        eps = torch.sign(z2)*RelaxedDivision.threshold*mask
        z2_inv = torch.reciprocal(z2 + eps)

        # Trick: do division as z1 * 1/z2. 1/z2 is also relaxed by adding the threshold as kinda eps
        res = z1 * z2_inv 

        ctx.save_for_backward(res, mask, z2)
        ctx.threshold = RelaxedDivision.threshold

        return res

    @staticmethod
    def backward(ctx, forward_output):

        res, mask, z2 = ctx.saved_tensors
        threshold = ctx.threshold

        steer_away_term = torch.maximum( threshold - z2, torch.zeros_like(z2) ) * mask

        grad = (forward_output * res) * mask

        return grad, grad + steer_away_term


# From https://github.com/martius-lab/EQL/blob/331f6023b195e5b93ba7ec191763a116f241d4b7/EQL-DIV-ICML-Python3/src/mlfg_final.py#L139

class BaseFuncLayer(nn.Module):
    """
    A `torch.nn` Module providing `basefunc[i](W*z1+b, [W*z2+b])`

    This layer applies available functions to all incoming features after applying a weight (and possibly bias).
    For functions with greater arity than 1, all combinations are tested. Currently only 2-ary functions are supported

    Supported functions and their indeces:
        1-ary:
            0: `cos(z)`
            1: `sin(z)`
        2-ary:
            0: `z1 * z2`
            1: `RelaxedDiv(z1, z2)` [ z1/z2 when z2 > threshold, 0 otherwise ]

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
            in_dim: int,
            n_per_base: int,
            # surprisingly the tf1 implementation sets bias to false
            bias: bool = True,
            basefuncs_1ary: List[int] = None,
            basefuncs_2ary: List[int] = None,
            l0_droprate_init: float = 0.5,
            l0_stdev_init: float = 1e-2,
            l0_BETA = 2/3,
            l0_GAMMA = -0.1,
            l0_ZETA = 1.1,
            l0_EPSILON = 1e-6,
            _validate_integrity: bool = False
        ):
        super().__init__()

        if basefuncs_1ary is None:
            basefuncs_1ary = [0, 1]
        if basefuncs_2ary is None:
            # Division disabled by default in the Martius TF implementation?
            #  leads to explosive extrapol error, see penalty and threshold tricks in Div-EQL (Sahoo16) and learnable threshold?
            basefuncs_2ary = [0]

        self.n_base_1ary = len(basefuncs_1ary)
        self.n_base_2ary = len(basefuncs_2ary)

        # NOTE: The idxs are parameters because they are used in a `where` operation and thus need to be in the same device as the inputs/parameters
        #        making them parameters makes it so that they are moved alongside the model on `.to` operations.
        self.fun_1ary_idx = nn.Parameter( torch.Tensor( basefuncs_1ary ).repeat(n_per_base * in_dim) , requires_grad=False)
        self.fun_2ary_idx = nn.Parameter( torch.Tensor( basefuncs_2ary ).repeat(n_per_base * in_dim) , requires_grad=False)
        
        self.in_dim  = in_dim
        self.n_per_base = n_per_base * in_dim
        self.n_w     = (self.n_base_1ary + 2 * self.n_base_2ary) * self.n_per_base
        self.out_dim = (self.n_base_1ary +     self.n_base_2ary) * self.n_per_base

        # L0
        self.l0_droprate_init = l0_droprate_init
        self.l0_stdev_init = l0_stdev_init
        self.BETA = l0_BETA
        self.GAMMA = l0_GAMMA
        self.ZETA = l0_ZETA
        self.EPSILON = l0_EPSILON

        self.qz_log_alpha = nn.Parameter(
            torch.distributions.normal.Normal(
                loc=np.log(1-self.l0_droprate_init) - np.log(self.l0_droprate_init),
                scale=1e-2,
            ).sample((in_dim, self.n_w)),
        )

        # Weight init
        self.W = nn.Parameter(
            torch.distributions.normal.Normal(
                loc=np.log(1-self.l0_droprate_init) - np.log(self.l0_droprate_init),
                scale=self.l0_stdev_init
            ).sample((self.in_dim, self.n_w)),
            requires_grad=True
        )

        self.b = nn.Parameter(
            0.1*torch.distributions.normal.Normal(
                loc=np.log(1-self.l0_droprate_init) - np.log(self.l0_droprate_init),
                scale=self.l0_stdev_init
            ).sample((self.n_w,)),
            requires_grad=bias
        )
        if not bias:
            self.b *= 0

        if _validate_integrity: self.__assert_deterministic_forward_is_consistent()

    @torch.no_grad()
    def __assert_deterministic_forward_is_consistent(self):
        """Asserts that the result by the deterministic forwards in Torch and the symbolic version in NP are equal
        """
        orig_dev = self.W.device
        self.to('cpu')

        symb_vars =  np.array([sp.Symbol(f'x{i+1}') for i in range(self.in_dim)])

        ins = torch.rand((1,self.in_dim), device=self.W.device)

        torch_det_fw = self.forward(ins, deterministic=True)[0].numpy(force=True)

        sp_det_fw = np.array([
            expr.subs({ x: inp.item() for x, inp in zip(symb_vars, ins[0]) }) for expr in self.symbolic_forward(symb_vars)
        ]).astype(np.float32)

        closeness = np.isclose(torch_det_fw, sp_det_fw)
        assert closeness.all(), \
        f'[INTERNAL]: Layer {self._get_name()} failed the consistency test\nin_dim={self.in_dim}, out_dim={self.out_dim}\nTorch={torch_det_fw}\nSP-NP={sp_det_fw}\nisclose={closeness}'

        self.to(orig_dev)
        return True

    def forward(self, input: torch.Tensor, deterministic: bool | None = None) -> torch.Tensor:

        deterministic = not self.training if deterministic is None else deterministic
        # 1. Apply wX+b
        W = (self.sample_weights() if not deterministic else self.deterministic_weights())
        node_input = input @ W + self.b

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
            # idx == 0 -> cos
            torch.eq(self.fun_1ary_idx, 0), torch.cos(z),
            # else -> sin
            torch.sin(z),
        )

        fn_2ary = torch.where(
            # idx == 0 -> *
            torch.eq(self.fun_2ary_idx, 0), z1*z2,
            #torch.where(
            #    # idx == 1 -> RelaxedDiv
            #    torch.eq(self.fun_2ary_idx, 1), RelaxedDivision.apply(z1, z2),
                # else -> id
                z1
            #)
        )

        # 4. Reshape into single column tensor output with f1ary, f2ary results
        out = torch.concat([fn_1ary, fn_2ary], dim=1)

        return out
    
    def symbolic_forward(self, input: np.ndarray) -> np.ndarray:
        """Apply a symbolic forward with SymPy variables. Should mirror `self.forward` but implemented with np/sympy operations rather than Torch
        """
        # 1. Apply wX+b
        node_input = input @ self.deterministic_weights().numpy(force=True) + self.b.numpy(force=True)

        # 2. Prepare tensors to apply basefuncs to
        z  = node_input[
            :self.n_per_base * self.n_base_1ary
        ]
        
        z1 = node_input[
            self.n_per_base*self.n_base_1ary : self.n_per_base * (self.n_base_1ary + self.n_base_2ary)
        ]
        z2 = node_input[
            self.n_per_base * (self.n_base_1ary + self.n_base_2ary) :
        ]

        # 3. Apply basefuncs
        fn_1ary = np.where(
            # idx == 0 -> cos 
            np.equal(self.fun_1ary_idx.numpy(force=True), 0), np.array([sp.cos(e) for e in z]),
            # else -> sin
            np.array([sp.sin(e) for e in z]),
        )

        fn_2ary = np.where(
            # idx == 0 -> *
            np.equal(self.fun_2ary_idx.numpy(force=True), 0), np.multiply(z1, z2),
            #np.where(
            #    # idx == 1 -> Div
            #    np.equal(self.fun_2ary_idx.numpy(force=True), 1), np.divide(z1, z2),
                # else -> identity
                z1
            #)
        )

        # 4. Reshape into single column tensor output with f1ary, f2ary results
        out = np.concatenate([fn_1ary, fn_2ary], axis=0)

        return out
    
    def quantile_concrete(self, u) -> torch.Tensor:
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1.0-u) + self.qz_log_alpha) / self.BETA)
        return y * (self.ZETA - self.GAMMA) + self.GAMMA
    
    def sample_uniform(self, shape) -> torch.Tensor:
        """Uniform random numbers for concrete distribution"""
        return torch.distributions.uniform.Uniform(low=self.EPSILON, high=1.0 - self.EPSILON).sample(shape).to(self.W.device)
    
    def sample_hard_concrete(self, batch_size) -> torch.Tensor:
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        eps = self.sample_uniform((batch_size, self.in_dim, self.n_w))
        z = self.quantile_concrete(eps)
        return torch.clamp(z, min=0, max=1)
    
    def sample_weights(self) -> torch.Tensor:
        """Create a mask for weights based on sampling from the concrete distribution"""
        z = self.quantile_concrete(self.sample_uniform((self.in_dim, self.n_w)))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W
    
    def hard_concrete_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.ZETA - self.GAMMA) + self.GAMMA, min=0.0, max=1.0)
    
    def deterministic_weights(self) -> torch.Tensor:
        """Deterministic values for weight matrix W based on Z's mean"""
        return self.W * (self.hard_concrete_mean() >= 0.5)
    
    def l0_loss(self)  -> float:
        """Compute a loss based on the expected number of non-zero weights using the mask that is obtained using the concrete distribution"""
        return torch.sum(
            torch.sigmoid(
                self.qz_log_alpha - self.BETA * np.log(-self.GAMMA / self.ZETA) 
            )
        )
  
class SkipBaseFuncLayer(BaseFuncLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim += self.in_dim

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out = super().forward(*args, **kwargs)

        return torch.concat([out, *args], dim=-1)
    
    def symbolic_forward(self, *args, **kwargs):
        out = super().symbolic_forward(*args, **kwargs)

        return np.concatenate([out, *args], axis=0)
    
class ResidualBaseFuncLayer(BaseFuncLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = torch.nn.Linear(self.out_dim, self.in_dim)
        self.l = nn.Parameter(
            torch.distributions.normal.Normal(
                loc=np.log(1-self.l0_droprate_init) - np.log(self.l0_droprate_init),
                scale=1e-2,
            ).sample((self.out_dim, self.in_dim)),
        )
        self.out_dim = self.in_dim

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out = out @ self.l
        return torch.add(out, *args)
    
    def symbolic_forward(self, *args, **kwargs):
        out = super().symbolic_forward(*args, **kwargs)
        out = out @ self.l.numpy(force=True)

        return np.add(out, *args)

class EQL(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            n_fglayers: int,
            n_per_base: int,
            bias: bool = True,
            basefuncs_1ary: List[int] | None = None,
            basefuncs_2ary: List[int] | None = None,
            _func_layer_cls: BaseFuncLayer = BaseFuncLayer,
        ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_fglayers = n_fglayers
        self.n_per_base = n_per_base
        self._func_layer_cls = _func_layer_cls

        last_in_dim = in_dim
        fglayers: List[BaseFuncLayer] = []
        for _ in range(n_fglayers):
            layer: BaseFuncLayer = _func_layer_cls(
                    in_dim=last_in_dim,
                    n_per_base=self.n_per_base,
                    bias=bias,
                    basefuncs_1ary=basefuncs_1ary,
                    basefuncs_2ary=basefuncs_2ary
            )
            fglayers.append(layer)
            last_in_dim = layer.out_dim

        self.base_func_hidden_layers: List[BaseFuncLayer] = nn.Sequential(
            *fglayers,
        )
        
        self.W = nn.Parameter(
            torch.distributions.uniform.Uniform(0, 1).sample((last_in_dim, self.out_dim)),
            requires_grad=True
        )

        self.b = nn.Parameter(
            torch.distributions.uniform.Uniform(0,1).sample((self.out_dim,)),
            requires_grad=bias
        )
        if not bias:
            self.b *= 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        
        out = self.base_func_hidden_layers(input)        
        out = out @ self.W + self.b

        return out
    
    def l0_loss(self) -> torch.Tensor:

        return torch.sum(
            torch.Tensor([ 
                sym_layer.l0_loss()
                for sym_layer in self.base_func_hidden_layers 
            ])
        )
    
    def prune_weights_under_threshold(self, threshold: float):

        with torch.no_grad():
            for p in self.parameters(recurse=True):
                if p.requires_grad:
                    mask = torch.gt(torch.abs(p), threshold)
                    p *= mask
                    if p.grad is not None:
                        p.grad *= mask
  
    def symbolic(self, threshold: float = 0, float_precision: int = 5):

        expr = in_vars = np.array([sp.Symbol(f'x{i+1}') for i in range(self.in_dim)])

        for layer in self.base_func_hidden_layers:
            expr = layer.symbolic_forward(expr)

        expr = expr @ self.W.numpy(force=True) + self.b.numpy(force=True)

        expr = [self.symbolic_prune(o, threshold) for o in expr]
        expr = [self.symbolic_round(o, float_precision) for o in expr]

        return expr[0], list(in_vars)
    
    def symbolic_round(_self, expr, num_digits: int):
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})
    
    def symbolic_prune(_self, expr, threshold: float):
        return expr.xreplace({n : (n if np.abs(n) > threshold else 0) for n in expr.atoms(sp.Number)})


class SkipEQL(EQL):

    def __init__(self,*args, **kwargs):
        kwargs.pop('_func_layer_cls',None)
        super().__init__(*args, **kwargs, _func_layer_cls=SkipBaseFuncLayer)

class ResidualEQL(EQL):
    
    def __init__(self,*args, **kwargs):
        kwargs.pop('_func_layer_cls',None)
        super().__init__(*args, **kwargs, _func_layer_cls=ResidualBaseFuncLayer)