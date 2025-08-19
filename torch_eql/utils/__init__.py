from typing import List, Callable, Tuple, Literal, Dict
import torch
from torch import nn
from prettytable import PrettyTable
# make available from this namespace?
#from .graphing import restore_model_from_report

AUTO_DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

def get_path_from_report(
    report: Dict,
    path: str,
    separator: str = '/'
) -> List:
    f"Extract a `{separator}`-separated path from report"

    res = report
    for subpath in path.split(separator):
        try:
            res = res[subpath]
        except:
            successful_traverse = path.rfind(subpath)
            raise ValueError(
                f'The indicated path {path} is not present in the report. Traversed {successful_traverse}/{subpath} <- error here'
            )
        
    return res


def restore_model_from_report(report: Dict, fn_in_dim: int) -> nn.Module:

    depth, width = report['model']['size'].split('x')
    model_type = report['model']['type']
    fn = report['result']['result_formula']

    # depack, move to `torch.` notation
    fn = fn.replace('sin', 'torch.sin').replace('cos', 'torch.cos')
    for i in range(fn_in_dim):
        fn = fn.replace(f'x{i+1}', f'X[:, {i}]')

    cls_source = (
        "class {model_type}(torch.nn.Module):\n" +
        "    n_fglayers={depth}\n" +
        "    n_per_base={width}\n" +
        "    def forward(self, X):\n" +
        "       return {fn}\n"
    ).format(depth=depth, width=width, fn=fn, model_type=model_type)
    
    locals = {}
    exec(cls_source, {'torch': torch}, locals)
    
    return locals[model_type]()

def report_n_parameters(model: nn.Module, trainable_only: bool = True, do_print: bool = True) -> float:
    table = PrettyTable(["Modules", "Parameters", "Size", "Device"], align='l', title=str(model.__class__.__name__))
    total_params = 0
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params, parameter.shape, parameter.device])
        total_params += params
    if do_print:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


def max_grad_change(
        m: nn.Module, 
        at: Literal['base_func_hidden_layers.qz_log_alpha', 'W', 'base_func_hidden_layers.W'] = 'W'
    ) -> float:
    """
    Returns the maximum absolute gradient change at the last layer
    """
    # shape of grad is in x out
    # grad_dim = 0: Magnitude of the output
    # grad_dim = 1: Max magnitude of the operator gradients

    grad_source = None
    grad_dim = 0
    if at == 'W':
        grad_source = m.get_parameter('W')
    if at == 'base_func_hidden_layers.W':
        grad_source = m.base_func_hidden_layers[-1].get_parameter('W')
        grad_dim = -1
    if at == 'base_func_hidden_layers.qz_log_alpha':
        grad_source = m.base_func_hidden_layers[-1].get_parameter('qz_log_alpha')
        grad_dim = -1

    if grad_source == None: raise RuntimeError(f'Invalid grad source {at}')

    return torch.max(
        torch.sum(torch.abs(
            grad_source.grad
        ), dim=grad_dim)
    ).item()