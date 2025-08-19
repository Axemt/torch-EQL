import torch
from torch import optim
from torch import nn
from torch_eql.EQL import EQL, SkipEQL, ResidualEQL
from torch_eql.EQL_with_spline import SkipEQLWithKan
from torch_eql.loss import LOSSES
from torch_eql.utils.generation import generate_disjoint_test, TEST_FNS, TestFNS
from torch_eql.utils import graphing
from torch_eql.utils.dissect import patch_into
from torch_eql.utils.heuristics import signal_from_tracked
from torch_eql.utils import report_n_parameters, max_grad_change, AUTO_DEVICE, get_path_from_report
from datetime import datetime
from math import isnan
import json
from datetime import datetime, timedelta
import argparse
from typing import Callable, Literal, List, Dict
import os
from warnings import warn

def run_fit(
    depth: int, width: int,
    model_cls: nn.Module = SkipEQL,
    device: str = AUTO_DEVICE,
    seed: int = 42,
    epochs: int = 1_000_00, 
    test_function: Callable[[torch.Tensor], torch.Tensor] = TestFNS.FG1a,
    train_domain: float = 1.0,
    train_size: int = int(1e4),
    test_domain: float = 3.0,
    test_size: int = int(1e5),
    optclass: optim.Optimizer = optim.RMSprop,
    l: nn.Module = LOSSES['l0mseloss'],
    step_strategy: Literal[1, 2, 3] = 3,
    heuristic_tracked: List[str] = [],
    heuristic_threshold: float = 0.95,
    heuristic_period: int | None = None,
    heuristic_evaluate_every: int = 1,
    heuristic_max_depth_growth: int = float('inf'),
    heuristic_max_depth_early_exit: bool = True,
    heuristic_growth_backoff_factor_increase: float = 1.0,
    heuristic_growth_backoff: int = 1000,
    heuristic_cold_run: bool = True,
    terseness: float = 0.1,
    project_dir: str = '.'
    ):

    torch.autograd.set_detect_anomaly(True)
    print(f'Using torch device `{device}`...')
    print(f'Seeding to {seed}...')
    torch.random.manual_seed(seed)

    print(f'Setting project path to {project_dir}...')
    if not os.path.isdir(project_dir): os.mkdir(project_dir)

    print('Generating samples...')

    (x_seens, y_seens), (x_unseens, y_unseens) = generate_disjoint_test(
        test_function,
        4,
        seen_interval=train_domain,
        unseen_interval=test_domain,
        seen_density=train_size,
        unseen_density=test_size,
        device=device
    )

    print('Building and verifying model...', end='', flush=True)
    model: EQL = model_cls(
        in_dim=4,
        out_dim=1,
        n_fglayers=depth,
        n_per_base=width
    ).to(device)
    model.train()
    patch_into(model_cls)

    print()
    report_n_parameters(model, trainable_only=True)

    print('OK')

    opt = optclass(model.parameters())

    minibatch = train_size

    n_batches = x_seens.shape[0]/minibatch
    assert x_seens.shape[0] % minibatch == 0, f'The number of samples is not divisible by batch size: samples: {x_seens.shape[0]}; batch size={minibatch}; n of batches={n_batches}'

    step = 1
    l0_lambda = terseness

    loss = l(L0_lambda=(0 if step_strategy > 1 else l0_lambda)).to(device)
    mse = nn.MSELoss()

    print(f'Training (step_strategy: {step_strategy})...')


    start = datetime.now()
    total_h_time = timedelta()
    total_train_time = timedelta()
    current_heuristic_growth_backoff = heuristic_growth_backoff

    report = {
        'model': {
            'type': str(model.__class__.__name__),
        },
        'function': str(test_function.__name__),
        'result': {},
        'time' : {},
        'device': {
            'device': device.type if device.type != 'cuda' else torch.cuda.get_device_name(device),
            'num_gpu_threads': torch.get_num_threads(),
            'seed': seed,
        },
        'train_settings': {
            'step_strategy': step_strategy,
            'terseness': terseness,
            'opt': str(opt.__class__.__name__),
            'loss': str(loss.__class__.__name__),
            'minibatch_size': minibatch,
            'epochs': None,
            'max_epochs': epochs,
            'heuristics': {
                'tracked': heuristic_tracked,
                'threshold': heuristic_threshold,
                'evaluate_every': heuristic_evaluate_every,
                'period': heuristic_period if heuristic_period is not None else 'auto',
                'cold_run': heuristic_cold_run,
                'starting_size': f'{depth}x{width}',
                'max_depth_growth': heuristic_max_depth_growth,
                'max_depth_early_exit': heuristic_max_depth_early_exit,
                'growth_points': [],
                'growth_backoff': heuristic_growth_backoff,
                'heuristic_growth_backoff_factor_increase': heuristic_growth_backoff_factor_increase,
                'live_heuristic': [],
            }
        },
        'loss' : {
            'interpolation_loss': [],
            'l0': []
        },
        'mse': {
            'interpolation_mse': [],
            'extrapolation_mse': [],
        },
        'grads': {
                'base_func_hidden_layers.W': [],
                'base_func_hidden_layers.qz_log_alpha': [],
                'W': []
        }
    }

    for i in range(epochs):

        t = datetime.now()


        if step_strategy >= 2 and i > (1/4)*epochs and step < 2:
        
            step = 2
            print('='*29,'Reached step 2, enabling L0 weight norm penalization...','='*29)
        
            loss.L0_lambda = l0_lambda
        
        if step_strategy >= 3 and i >= (19/20)*epochs and step < 3:
        
            step = 3
            print('='*29,'Reached step 3, enabling pruning under threshold weights...','='*29)
            
            loss.L0_lambda = 0

        epoch_interpolation_loss = 0
        epoch_interpolation_l0_loss = 0
        epoch_interpolation_mse = 0
        epoch_extrapolation_mse = 0
        train_start = datetime.now()
        for _minibatch, (x_train, y_train) in enumerate(zip(
            x_seens.reshape((x_seens.shape[0]//minibatch, minibatch, 4)), 
            y_seens.reshape((y_seens.shape[0]//minibatch, minibatch, 1))
        )):

            out_one = model(x_train)

            out_loss = loss(model, out_one, y_train)

            epoch_interpolation_loss += out_loss.item()
            epoch_interpolation_l0_loss += terseness * model.l0_loss().item()

            out_loss.backward()

            
            if step_strategy >= 3 and step == 3:
            ## Routinely kill small weights in every step 3 epoch
            ## from https://github.com/martius-lab/EQL_Tensorflow/blob/95f6de7e9e4494fd838fbabf3622f3d76623fe2f/EQL_Layer_tf.py#L98
                model.prune_weights_under_threshold(threshold=terseness)
            

            # FIXME: This is very wrong in cases where the minibatch is not equal to a full batch. Agg if that's a case we care about
            for g in report['grads'].keys():

                report['grads'][g].append( max_grad_change(model, at=g) )

            opt.step()
            opt.zero_grad()

        total_train_time += datetime.now() - train_start

        with torch.no_grad():

            model.eval()
            y_hat_inter = model(x_seens).flatten()
            y_hat_extra = model(x_unseens).flatten()

            epoch_interpolation_mse += mse(y_seens, y_hat_inter)
            epoch_extrapolation_mse += mse(y_unseens, y_hat_extra)

            model.train()

        report['loss']['interpolation_loss'].append(epoch_interpolation_loss)
        report['loss']['l0'].append(epoch_interpolation_l0_loss)
        report['mse']['interpolation_mse'].append(epoch_interpolation_mse.cpu().item())
        report['mse']['extrapolation_mse'].append(epoch_extrapolation_mse.cpu().item())

        if i % heuristic_evaluate_every == 0 and heuristic_tracked != []:
            h_start = datetime.now()

            h = signal_from_tracked(
                [
                    get_path_from_report(report, h) for h in heuristic_tracked
                ],
                epochs
            )

            if not heuristic_cold_run and h >= heuristic_threshold and i > ([0]+report["train_settings"]["heuristics"]["growth_points"])[-1]+current_heuristic_growth_backoff:

                if model.n_fglayers >= heuristic_max_depth_growth:
                    print(f'\tTriggered growth signal but hit the growth limit of {heuristic_max_depth_growth} layers')
                    if heuristic_max_depth_early_exit:
                        print(f'Finishing train loop early...')
                        break
                else:
                    print(f'\tTriggered growth signal with h={h}\nGrowing depthwise...', end='', flush=True)
                    model.grow_depthwise(opt=opt)
                    print('OK')
                    report["train_settings"]["heuristics"]["growth_points"].append(i)
                    current_heuristic_growth_backoff *= heuristic_growth_backoff_factor_increase

            total_h_time += datetime.now() - h_start
            report['train_settings']['heuristics']['live_heuristic'].append(h)

        print(
            (f"[Step {step}]" if step_strategy > 1 else "") +
            f'[{model.n_fglayers}x{model.n_per_base} {str(model.__class__.__name__)}]' +
            f'[Total: {datetime.now() - start}][Epoch: {datetime.now() - t}][{((i+1)/epochs) * 100 :.3f} %] @ Epoch {i+1 :<5}; ' +
            f'{loss.__class__.__name__}={epoch_interpolation_loss : >.10f} L0={epoch_interpolation_l0_loss : >.10f}; ' +
            f'MSEs: inter={epoch_interpolation_mse : .5f}; extra={epoch_extrapolation_mse : .5f}' + 
            (
                    (
                        f'; In backoff ({100-(([0]+report["train_settings"]["heuristics"]["growth_points"])[-1]+current_heuristic_growth_backoff-i)/current_heuristic_growth_backoff * 100 :.3f}%)' 
                        if (i <= ([0]+report["train_settings"]["heuristics"]["growth_points"])[-1]+current_heuristic_growth_backoff) 
                        else f'; Signal: {report["train_settings"]["heuristics"]["live_heuristic"][-1]:.3f} Thr@{heuristic_threshold};'
                    ) if (heuristic_tracked != [] and not heuristic_cold_run) else ''
            )
        )
        
        if isnan(epoch_interpolation_mse): 
            print('MSE went to NaN! - Update failed!')
            break

    end = datetime.now()
    print('Generating symbolic formula...', end='', flush=True)

    formula_extraction_time = datetime.now()
    formula = model.symbolic(threshold=0)[0]
    formula_extraction_time = datetime.now() - formula_extraction_time
    print('OK')

    with open(f'{project_dir}/formula.txt', 'w') as f:
        f.write(str(formula))

    print('Writing training report...', end='', flush=True)
    report['train_settings']['epochs'] = len(report['loss']['interpolation_loss'])
    report['time'] = {
            'start': str(start),
            'end': str(end),
            'total_time': str(end - start),
            'train_time': str(total_train_time),
            'heuristic_time': str(total_h_time),
            'formula_extraction_time': str(formula_extraction_time)
    }
    report['result'] = {
            'result_formula': str(
                model.symbolic_round(
                    model.symbolic_prune(formula, threshold=terseness),
                    num_digits=5
                )),
            'unpruned_formula': str(formula),
    }
    report['model']['size'] = f'{model.n_fglayers}x{model.n_per_base}'
    report['model']['n_params'] = report_n_parameters(model, do_print=False)

    with open(
            f'{project_dir}/{str(test_function.__name__)}-{str(model.__class__.__name__)}' +
            f'-{model.n_fglayers}x{model.n_per_base}' +
            f'-e-{epochs}-b-{minibatch}.json', 
            'w'
        ) as f:

        json.dump(
            report,
            f,
            indent=1
        )
    print('OK')
    print('Plotting...', end='', flush=True)

    graphing.plot_function(
        test_function=test_function,
        fn_in_dims=model.in_dim,
        train_domain=train_domain,
        test_domain=test_domain,
        model=model,
        save_dir=project_dir,
        report=report
    )

    graphing.plot_loss(
        report, 
        save_dir=project_dir
    )

    graphing.plot_grads(
        report,
        save_dir=project_dir
    )

    for g in report['train_settings']['heuristics']['tracked']:
        graphing.plot_oracle_heuristic_signal(
            report,
            g,
            save_dir=project_dir
        )

    if len(heuristic_tracked) > 0:
        graphing.plot_live_heuristic_signal(
            report,
            save_dir=project_dir
        )

    print('OK')
    


def cli_call(*args, **kwargs):

    parser = argparse.ArgumentParser()


    MODEL_CLS = {
        'eql': EQL,
        'skip': SkipEQL,
        'residual': ResidualEQL
    }


    OPTS = {
        'rmsprop': optim.RMSprop,
        'adam': optim.Adam
    }

    parser.add_argument(
        'size',
        help='The size of the network. In the form AxB, where A is the depth, and B is the width',
        type=str
    )

    parser.add_argument(
        '--eql_type',
        help=f'The EQL type. From {list(MODEL_CLS.keys())}',
        type=str,
        default='eql'
    )

    parser.add_argument(
        '--seed',
        help='The RNG seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='The number of training epochs',
        default=1_000_0#00
    )

    parser.add_argument(
        '-tfn',
        '--test_function',
        type=str,
        help=f'A test function, from {list(TEST_FNS.keys())}',
        default='fg1a'
    )

    parser.add_argument(
        '--train_domain',
        help='The train domain. Given as [-x, x].',
        type=float,
        default=1
    )

    parser.add_argument(
        '--train_size',
        help='The number of train samples',
        type=int,
        default=int(1e4)
    )

    parser.add_argument(
        '--test_domain',
        help='The test domain. Given as [-x, x].',
        type=float,
        default=3
    )

    parser.add_argument(
        '--test_size',
        help='The number of test samples',
        type=int,
        default=int(1e4)
    )

    parser.add_argument(
        '-o',
        '--optimizer',
        help=f'The optimizer. From {list(OPTS.keys())}',
        default='rmsprop'
    )

    parser.add_argument(
        '-l',
        '--loss',
        help=f'The loss function to use, in lower case. From {list(LOSSES.keys())}',
        default='l0mseloss'
    )

    parser.add_argument(
        '--step_strategy',
        help='The stepped strategy level',
        type=int,
        default=3
    )

    parser.add_argument(
        '--heuristic_tracked',
        help='Enable heuristic growth for the indicated metric paths in the report. Comma separated',
        type=str,
        default=''
    )

    parser.add_argument(
        '--heuristic_threshold',
        help='The heuristic signal threshold',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--heuristic_period',
        help='Set a fixed period for heuristic window evaluation',
        type=int,
        default=None
    )

    parser.add_argument(
        '--heuristic_evaluate_every',
        help='How many epochs to wait between heuristic evaluations',
        type=int,
        default=1
    )

    parser.add_argument(
        '--heuristic_cold_run',
        help='Disables heuristic growth but still calculates and tracks the heuristic metric',
        action='store_true'
    )

    parser.add_argument(
        '--heuristic_growth_backoff',
        help='Sets a cooldown period between successive network growths',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--heuristic_growth_backoff_factor_increase',
        help='Sets a factor to increase growth cooldown on every successful growth',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--heuristic_max_depth_growth',
        help='Limit the depth growth via heuristics',
        default=float('inf'),
        type=int
    )

    parser.add_argument(
        '--heuristic_max_depth_early_exit',
        help='Terminate early if a growth signal is triggered when `heuristic_max_depth_growth` has already been reached',
        action='store_true'
    )

    parser.add_argument(
        '-t',
        '--terseness',
        help='The L0 penalty for parameter regularization',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--project_dir',
        help='The directory to save all outputs to',
        default='.'
    )

    args = parser.parse_args()

    depth, width = args.size.lower().split('x')

    run_fit(
        int(depth), int(width),
        model_cls=MODEL_CLS[args.eql_type.lower()],
        seed=args.seed,
        epochs=args.epochs,
        test_function=TEST_FNS[args.test_function.lower()],
        train_domain=args.train_domain,
        train_size=args.train_size,
        test_domain=args.test_domain,
        test_size=args.test_size,
        optclass=OPTS[args.optimizer.lower()],
        step_strategy=args.step_strategy,
        heuristic_tracked=[ gn.strip() for gn in args.heuristic_tracked.strip().split(',') if gn != '' ],
        heuristic_threshold=args.heuristic_threshold,
        heuristic_period=args.heuristic_period,
        heuristic_evaluate_every=args.heuristic_evaluate_every,
        heuristic_max_depth_growth=args.heuristic_max_depth_growth,
        heuristic_max_depth_early_exit=args.heuristic_max_depth_early_exit,
        heuristic_growth_backoff=args.heuristic_growth_backoff,
        heuristic_growth_backoff_factor_increase=args.heuristic_growth_backoff_factor_increase,
        heuristic_cold_run=args.heuristic_cold_run,
        terseness=args.terseness,
        project_dir=args.project_dir
    )

if __name__ == '__main__':
    cli_call()