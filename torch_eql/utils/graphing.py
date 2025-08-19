from matplotlib import pyplot as plt
from typing import Dict, List, Callable
from ..utils.generation import TestFNS
from ..utils import AUTO_DEVICE, restore_model_from_report, get_path_from_report
from ..utils.heuristics import detect_periodicity, all_signal_tendencies
from torch import nn
import torch
import numpy as np
import pandas as pd
from math import ceil

def plot_function(
        test_function: TestFNS.TestFNSignature,
        fn_in_dims: int,
        train_domain: List[float] | float, 
        test_domain: List[float] | float, 
        model: nn.Module | None = None,
        save_dir: str | None = None,
        report: Dict = {},
        device = AUTO_DEVICE
    ):

    if isinstance(test_domain, (float, int)):
        test_domain = [-test_domain, test_domain]

    if isinstance(train_domain, (float, int)):
        train_domain = [-train_domain, train_domain]

    x_seens   = torch.arange(train_domain[0], train_domain[1], 0.0001).reshape((-1,1)).repeat(fn_in_dims,fn_in_dims).to('cpu')
    x_unseens = torch.arange(test_domain[0], test_domain[1], 0.0001).reshape((-1,1)).repeat(fn_in_dims,fn_in_dims).to('cpu')
    y_seens    = test_function(x_seens).to('cpu')
    y_unseens  = test_function(x_unseens).to('cpu')
    seen_error   = report.get('mse', {}).get('interpolation_mse', [float('nan')])[-1]
    unseen_error = report.get('mse', {}).get('extrapolation_mse', [float('nan')])[-1]

    loss_name = report.get('train_settings', {}).get('loss', '???')

    if model is None and 'result_formula' in report.get('result', {}):
        print('Restoring model from report...')
        model = restore_model_from_report(report, fn_in_dims)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if model is not None:

        with torch.no_grad():
            model.training = False        
            y_hat_seens = model(x_seens.to(device)).to('cpu').tolist()
            y_hat_unseens = model(x_unseens.to(device)).to('cpu').tolist()

            ax.scatter(x_unseens[:, 0], y_hat_unseens, 1, label=f'{str(model.__class__.__name__)} {loss_name} unseens')
            ax.scatter(x_seens[:, 0], y_hat_seens, 1,     label=f'{str(model.__class__.__name__)} {loss_name} seen', alpha=0.8)
            ax.set_title(f'MSE Error: seen ±{seen_error : .5f}; unseen ±{unseen_error : .5f}')

    suptitle = f"{test_function.__name__} " 
    if model is not None: suptitle += f"{str(model.__class__.__name__)} @ Epoch {report.get('train_settings', {}).get('epochs', '???')}; {model.n_fglayers}x{model.n_per_base} ($\lambda_{{L0}}$ {report.get('train_settings', {}).get('terseness', '???')})"
    fig.suptitle(suptitle)

    x = torch.concat([x_seens,x_unseens])[:,0].cpu()
    x_seens   =   x_seens[:,0].cpu()
    x_unseens = x_unseens[:,0].cpu()
    y_true = torch.concat([y_seens, y_unseens]).tolist()

    ax.scatter(x, y_true, 1, label='true val', alpha=0.8, color='b')

    ax.axvspan(train_domain[0],train_domain[1], facecolor='0.8', alpha=0.5)

    ax.set_ylabel('y')
    ax.set_xlabel( ''.join((f'x{i+1}=' for i in range(fn_in_dims))) + 'x' )
    ax.set_ylim(min(y_true), max(y_true))
    ax.set_xlim(test_domain[0], test_domain[1])

    fig.tight_layout()
    #fig.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    if save_dir is not None: plt.savefig(f'{save_dir}/eql perf', bbox_inches='tight')
    return fig

def _add_strategy_markers(report, ax):
        
    step_strategy = report.get('train_settings', {}).get('step_strategy', 1)
    epochs = report['train_settings']['epochs']


    if step_strategy >= 2:
        ax.axvspan(0, 1/4*epochs, label='No decay', alpha=0.3, color='lightgray')
        step_2_end = (19/20*epochs) if step_strategy >=3 else epochs
        ax.axvspan(1/4*epochs, step_2_end, label=f'L0 penalty', alpha=0.3, color='lightgreen')
        ax.axvline(1/4*epochs, color='k', linestyle='--')

        if step_strategy >= 3:
            ax.axvspan(19/20*epochs, epochs, label=f'L0 norm enforcement', alpha=0.3, color='lightblue')
            ax.axvline(19/20*epochs, color='k', linestyle='--')

def _add_network_growths(report, ax):

    growth_points = report['train_settings'].get('heuristics', {}).get('growth_points', [])
    growth_backoff = report['train_settings'].get('heuristics', {}).get('growth_backoff', 0)
    growth_backoff_factor_increase = report['train_settings'].get('heuristics', {}).get('heuristic_growth_backoff_factor_increase', 1)
    is_cold = report['train_settings'].get('heuristics', {}).get('cold_run', True)

    if not is_cold:
        ax.axvspan(0, growth_backoff, label='Growth Backoff', color='r', alpha=0.1)
    if len(growth_points) > 0:

        for gp in growth_points:

            growth_backoff *= growth_backoff_factor_increase
            ax.axvline(gp, color='r', label='Network growth')
            ax.axvspan(gp, gp+growth_backoff, label='Growth Backoff', color='r', alpha=0.1)

def plot_loss(
    report: Dict,
    save_dir: str | None = None,
    ):

    model_size = report.get('model', {}).get('size', "?x?")
    model_class = report.get('model', {}).get('type', '???')
    minibatch = report.get('train_settings', {}).get('minibatch_size', "???")
    l0_lambda = report.get('train_settings', {}).get('terseness', '???')
    optimizer = report.get('train_settings', {}).get('opt', '???')
    loss = report.get('train_settings', {}).get('loss', "???")

    epochs = report['train_settings']['epochs']
    interpolation_losses = report['loss'].get('interpolation_loss', [])
    interpolation_l0_losses = report['loss'].get('l0', [])
    interpolation_mses = report['mse']['interpolation_mse']
    extrapolation_mses = report['mse']['extrapolation_mse']


    best_interpol_loss_epoch = np.argmin(interpolation_losses)
    best_interpol_mse_epoch = np.argmin(interpolation_mses)
    best_extrapol_mse_epoch = np.argmin(extrapolation_mses)


    fig = plt.figure()
    fig.suptitle(f'Performance metrics. {model_size} {model_class}\nbatch size {minibatch}; $\lambda_{{L0}}$={l0_lambda}; {optimizer} optimizer.')
    ax_loss = fig.add_subplot(111)


    ax_loss.scatter(best_interpol_loss_epoch, interpolation_losses[best_interpol_loss_epoch], label='best loss', marker='x', s=50, color='k')
    ax_loss.set_xlim(right=epochs, left=0)

    ax_loss.plot(range(len(interpolation_losses)), interpolation_losses, color='k', label=f'Loss')
    ax_loss.plot(range(len(interpolation_l0_losses)), interpolation_l0_losses, color='red', label=f'L0 Loss')

    _add_strategy_markers(report, ax_loss)
    _add_network_growths(report, ax_loss)

    ax_loss.set_ylabel(f'{loss} Loss')
    ax_loss.set_xlabel('Epoch')

    ax_mse = ax_loss.twinx()


    interpolation_mses = np.log10(interpolation_mses)
    extrapolation_mses = np.log10(extrapolation_mses)

    ax_mse.scatter(best_interpol_mse_epoch, interpolation_mses[best_interpol_mse_epoch], label='best interpolation MSE', marker='x', s=50, color='red')
    ax_mse.scatter(best_extrapol_mse_epoch, extrapolation_mses[best_extrapol_mse_epoch], label='best extrapolation MSE', marker='o', s=50, color='purple')

    ax_mse.plot(range(len(interpolation_mses)), interpolation_mses, label='interpolation MSE', color='orange')
    ax_mse.plot(range(len(extrapolation_mses)), extrapolation_mses, label='extrapolation MSE', color='green')

    ax_mse.set_ylabel('log10-MSE Error')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    if save_dir is not None: fig.savefig(f'{save_dir}/eql training loss', bbox_inches='tight')
    return fig

def plot_grads(
    report: Dict,
    save_dir: str | None = None 
):
    
    model_size = report.get('model', {}).get('size', "?x?")
    model_class = report.get('model', {}).get('type', '???')
    l0_lambda = report.get('train_settings', {}).get('terseness', '???')
    
    epochs = report['train_settings']['epochs']
    grad_out_W = report['grads']['W']
    grad_last_basefunc_W = report['grads']['base_func_hidden_layers.W']
    grad_last_basefunc_qz = report['grads']['base_func_hidden_layers.qz_log_alpha']

    interpolation_losses = report['loss']['interpolation_loss']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(grad_out_W)), np.log(grad_out_W), color='blue', label='Out W Grad magnitude')
    ax.plot(range(len(grad_last_basefunc_W)), np.log(grad_last_basefunc_W), color='green', label='Last W Grad magnitude')
    ax.plot(range(len(grad_last_basefunc_qz)), np.log(grad_last_basefunc_qz), color='orange', label='Last QZ Grad magnitude')
    ax.set_ylabel('Max gradient magnitude')
    ax.set_xlabel('Epoch')
    loss_ax = ax.twinx()
    loss_ax.plot(range(len(grad_out_W)), interpolation_losses, color='red', label='loss')
    loss_ax.set_ylabel('loss')
    plt.title('Last-layer max gradient magnitude by loss')
    fig.suptitle(f'{model_class} @ Epoch {epochs}; {model_size} (terseness {l0_lambda})')
    fig.tight_layout()
    
    _add_strategy_markers(report, loss_ax)
    _add_network_growths(report, loss_ax)

    fig.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    if save_dir is not None: fig.savefig(f'{save_dir}/gradient gradmag by loss', bbox_inches='tight')
    return fig

def plot_live_heuristic_signal(
    report: Dict,
    save_dir: str | None = None
):
    
    trigger_threshold = report['train_settings']['heuristics']['threshold']
    total_epochs = report['train_settings']['epochs']
    tracked = report['train_settings']['heuristics']['tracked']
    fig = plt.figure(figsize=(10,5))
    ax_grad = fig.add_subplot(111)
    ax_grad.set_ylabel('Log-metrics')
    _add_strategy_markers(report, ax_grad)
    _add_network_growths(report, ax_grad)
    x = np.array(range(0,total_epochs))
    
    for tracked_path in tracked:
        g = np.log(get_path_from_report(report, tracked_path))
        ax_grad.plot(x, g, alpha=0.5, label=f'log-{tracked_path}')
    
    ax_grad.set_ylabel(f'log-metrics')
    ax_grad.set_xlabel('Epochs')

    ax_signal = ax_grad.twinx()
    ax_signal.set_ylabel('Heuristic Signal')
    ax_signal.set_ylabel('Signal')
    #ax_signal.set_ylim(-0.2, 1.2)


    ax_signal.hlines([trigger_threshold], 0, total_epochs, label='trigger', color='k')


    signal = np.array(report['train_settings']['heuristics']['live_heuristic'])
    heuristic_evaluate_every = report['train_settings']['heuristics'].get('evaluate_every', 1)

    triggerers = signal >= trigger_threshold 
    x = np.array(range(0,total_epochs, heuristic_evaluate_every))
    ax_signal.plot(x, signal, alpha=0.8, label='signal', color='r')
    ax_signal.scatter(x[triggerers], signal[triggerers], color='r')

    fig.suptitle(f'Live heuristic (threshold @ {trigger_threshold})')
    plt.title('Signal sources: ' + ' &'.join(tracked))
    fig.legend(loc='lower left', bbox_to_anchor=(0.5, -0.4))
    if save_dir is not None: fig.savefig(f'{save_dir}/live heuristic', bbox_inches='tight')
    return fig


def plot_oracle_heuristic_signal(
    report: Dict, 
    metric: str,
    band: int | None = None,
    period: int | None = None,
    at: int | None = None,
    save_dir: str | None = None
):
    g = np.log(get_path_from_report(report, metric))
    if at is not None:
        g = g[:at]

    s = pd.Series(g)
    total_epochs = report['train_settings']['epochs']
    trigger_threshold = report['train_settings']['heuristics']['threshold']

    period = detect_periodicity(s) if period is None else period
    band = ceil(2*np.log10(total_epochs)) if band is None else band
    
    signal_band = period*band

    fig = plt.figure(figsize=(10,5))
    ax_grad = fig.add_subplot(111)
    _add_strategy_markers(report, ax_grad)
    _add_network_growths(report, ax_grad)

    ma = s.rolling(period).median()
    ax_grad.plot(list(range(len(s))), ma.values, alpha=0.7, label='median')
    ax_grad.plot(list(range(len(s))), g, alpha=0.5, label='metric')
    ax_grad.set_ylabel(f'log-{metric} and log-moving metric')
    ax_grad.set_xlabel('Epochs')

    ax_signal = ax_grad.twinx()
    ax_signal.set_ylabel('Signal')
    ax_signal.set_ylim(-0.2, 1.2)


    ax_signal.hlines([trigger_threshold], 0, len(s), label='trigger', color='k')

    x = np.array(range(len(s)))

    a2_signal = all_signal_tendencies(ma, signal_band)
    a2_triggerers = a2_signal >= trigger_threshold 
    ax_signal.plot(x, a2_signal, alpha=0.8, label='mean signal', color='r')
    ax_signal.scatter(x[a2_triggerers], a2_signal[a2_triggerers], color='r')

    fig.suptitle(f'{metric}: Signals and EMs from {period}-frame window; {signal_band} signal band (x{band}); Trigger @ {trigger_threshold}')
    if save_dir is not None: fig.savefig(f'{save_dir}/signal oracle {metric[ metric.rfind("/")+1: ]}.png', bbox_inches='tight')
    return fig

    
def plot_oracle_signal_method_comparison(
    report: Dict, 
    grad_name: str,
    band: int | None = None,
    period: int | None = None,
    at: int | None = None,
    trigger_threshold: float = 0.95,
    save_dir: str | None = None
):
    
    g = np.log(report['grads'][grad_name])
    if at is not None:
        g = g[:at]

    s = pd.Series(g)
    total_epochs = report['train_settings']['epochs']

    period = detect_periodicity(s) if period is None else period
    band = ceil(2*np.log10(total_epochs)) if band is None else band
    
    signal_band = period*band

    fig = plt.figure(figsize=(10,5))
    ax_a = fig.add_subplot(121)
    _add_strategy_markers(report, ax_a)
    _add_network_growths(report, ax_a)

    ax_b = fig.add_subplot(122)
    _add_strategy_markers(report, ax_b)
    _add_network_growths(report, ax_b)

    ax_a.plot(list(range(len(s))), s.rolling(period).mean(), alpha=0.7, label='mean')
    ax_b.plot(list(range(len(s))), s.rolling(period).median(), alpha=0.7, label='median')

    ax_a.plot(list(range(len(s))), g, alpha=0.5, label='grad')
    ax_b.plot(list(range(len(s))), g, alpha=0.5, label='grad')

    ax_a2 = ax_a.twinx()
    ax_a2.set_ylim(-1, 2)
    ax_b2 = ax_b.twinx()
    ax_b2.set_ylim(-1, 2)


    ax_a2.hlines([trigger_threshold], 0, len(s), label='trigger', color='k')
    ax_b2.hlines([trigger_threshold], 0, len(s), label='trigger', color='k')

    x = np.array(range(len(s)))

    a2_signal = all_signal_tendencies(s.rolling(period).mean(), signal_band)
    a2_triggerers = a2_signal >= trigger_threshold 
    ax_a2.plot(x, a2_signal, alpha=0.8, label='mean signal', color='r')
    ax_a2.scatter(x[a2_triggerers], a2_signal[a2_triggerers], color='r')

    b2_signal = all_signal_tendencies(s.rolling(period).median(), signal_band)
    b2_triggerers = b2_signal >= trigger_threshold
    ax_b2.plot(x, b2_signal, alpha=0.8, label='median signal', color='r')
    ax_b2.scatter(x[b2_triggerers], b2_signal[b2_triggerers], color='r')

    ax_a.set_title('Mean-based')
    ax_b.set_title('Median-based')

    fig.suptitle(f'{grad_name}: Signals and EMs from {period}-frame window; {signal_band} signal band (x{band}); Trigger @ {trigger_threshold}')
    if save_dir is not None: fig.savefig(f'{save_dir}/{grad_name} signal method comparison', bbox_inches='tight')
    return fig
