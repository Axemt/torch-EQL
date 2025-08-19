from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from warnings import warn
import numpy as np

def grow_depthwise(self: nn.Module, opt: Optimizer | None = None, initial_connection_value: float = 1.0, **kwargs) -> nn.Module:
    """
    Grows the network depthwise, retaining current matrix values

    Kwargs:
        opt: A `torch.optim` Optimizer on which to update the tracked parameters with the new parameters
        initial_connection_value: A float to use as the initial value for the connection on skip, if there is any
        **kwargs: Arguments for the BaseFuncLayer class
    """
    should_update_opt = opt is not None and self.W.requires_grad
    if not should_update_opt:
        warn(
            'No optimizer was passed. If you are training the grown model, ' +
            'the new parameters are not currently tracked by the original optimizer. ' +
            'To do so, pass `opt=optimizer` to the grow function'
        )
    
    device = self.W.device
    pre_connect = self.base_func_hidden_layers[-1]

    new_depth_layer = self._func_layer_cls(
        pre_connect.out_dim,
        pre_connect.n_per_base // pre_connect.in_dim,
        **kwargs
    ).to(device)

    requires_w_extension = self.W.shape[0] != new_depth_layer.n_w

    if requires_w_extension:
        # Probablemente un skip: por tanto, W de salida no sera
        #  igual que la salida de la nueva ultima capa, porque hay una concatenacion
        # Solucion: Extendemos la matriz W, conservando los pesos

        new_W = nn.Parameter(F.pad(
            self.W, 
            # Do not grow on first dim
            (
                0, 0,
                # and then grow on the left side of last dim by difference of input shape
                0, new_depth_layer.out_dim - self.W.shape[0]
            ),
            # initialized to `initial_connection_value`
            mode='constant', value=initial_connection_value),
            requires_grad=self.W.requires_grad
        ).to(device)
        
        assert new_W.shape[0] == new_depth_layer.out_dim, '[INTERNAL]: The connection failed'
        self.W = new_W

    self.base_func_hidden_layers.append(
        new_depth_layer
    )
    self.n_fglayers += 1

    if should_update_opt:
        opt.add_param_group({'params': new_depth_layer.parameters()})

        if requires_w_extension:
            opt.add_param_group({'params': new_W})

def patch_into(cls):
    """
    Patches a class to include architecture modification functions

    cls: The class or instance we wish to add methods to
    """

    cls.grow_depthwise = grow_depthwise