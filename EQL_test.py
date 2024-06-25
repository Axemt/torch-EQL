import torch
from torch import optim
from torch import nn
from torch_EQL.EQL import EQL
from torch_EQL.loss import L1MSELoss
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

def FG1a(x):
    x1, x2, x3, x4 = x
    return (torch.sin(x1*torch.pi)+torch.sin(x2*torch.pi*2+torch.pi/8)+x2-x3*x4)/3.0

eql = EQL(
    in_features=4,
    n_fglayers=2,
    n_per_base=4,
).to('mps')

minibatch = 64
epochs = 100

seen_range = 3
granularity = 0.001

def modify_optim_with_decay(optimizer, weight_decay):
    for i in range(len(optimizer.param_groups)):

        if 'weight_decay' in optimizer.param_groups[i]:
            optimizer.param_groups[i]['weight_decay'] = weight_decay

optimizer = optim.Adam(eql.parameters(), lr=0.01, weight_decay=0)
loss = L1MSELoss()

print('Generating samples...')
unseens = {}
train = {}

range_ends = seen_range + seen_range*2
for v in tqdm(torch.arange(-range_ends, range_ends, granularity)):

    x = torch.Tensor([v, v, v, 0.1*v]).to('mps')
    y = FG1a(x)
    #gaussian_noise = (torch.randn((1,)) * torch.sqrt(torch.Tensor([granularity]))).to('mps')
    gaussian_noise = 0
    y = (y + gaussian_noise).unsqueeze(dim=0)
    x = x.unsqueeze(dim=0)

    if v >= -seen_range and v < seen_range: 
        train[x] = y
    else:
        unseens[x] = y

y_extrapol = torch.Tensor(list(unseens.values())).to('mps').unsqueeze(dim=1)
x_extrapol = torch.Tensor([list(e[0]) for e in unseens.keys()]).to('mps')

step = 1

print('Training...')
extrapolation_losses = []
interpolation_losses = []
for i in range(epochs):

    if i > (1/4)*epochs and step < 2:

        step = 2
        print('Reached step 2, enabling weight decay...')
        optimizer.step()
        optimizer.zero_grad()

        modify_optim_with_decay(optimizer, 1e-7)

    if i > (19/20)*epochs and step < 3:

        step = 3
        print('Reached step 3, enabling L0 norm...')

        for p in eql.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(
                    # L0 norm: For low impact weights |a|<0.001, a=0
                    lambda p: p.set_(torch.where(
                        torch.lt(torch.abs(p), 0.001),
                        0,
                        p
                    ))
                )

        optimizer.step()
        optimizer.zero_grad()
        modify_optim_with_decay(optimizer, 0)


    eql_seen_vals = []

    epoch_interpolation_loss = 0
    for j, (x_train, y_train) in enumerate(train.items()):


        out = eql(x_train)
        eql_seen_vals.append(out.item())

        output_loss = loss(eql.parameters(), out, y_train.unsqueeze(dim=1))
        epoch_interpolation_loss += output_loss.item()

        output_loss.backward()

        if (j+1) % minibatch == 0:

            optimizer.step()
            optimizer.zero_grad()


    interpolation_losses.append(epoch_interpolation_loss)
    
    with torch.no_grad():
        out_extrapol = eql(x_extrapol)
        epoch_extrapolation_loss = loss(eql.parameters(), out_extrapol, y_extrapol).item()
        extrapolation_losses.append(epoch_extrapolation_loss)


    print(f'[Step {step}]: Epoch {i+1 :<4}/{epochs :<4}: interpol loss={epoch_interpolation_loss : >.6f}; extrapol loss={epoch_extrapolation_loss : >.6f}')

print('Graphing...')

xs_seen = [ t[0][0].cpu() for t in train.keys()]
xs_unseens = [ t[0][0].cpu() for t in unseens.keys() ]
ys = [ v[0].cpu() for v in train.values()]  + [ v[0].cpu() for v in unseens.values() ]

plt.title(f'EQL @ Epoch {i+1}; {eql.n_fglayers}x{eql.n_per_base} EQL')
plt.scatter(xs_seen + xs_unseens, ys, 1, label='true val', alpha=0.8)
plt.scatter(xs_seen, eql_seen_vals, 1, label='eql seen', alpha=0.8)

eql.training = False
y_hat_unseens = [ eql(x_unseen).item() for x_unseen in unseens.keys() ]

plt.scatter(xs_unseens, y_hat_unseens, 1, label='eql unseens')
plt.axvspan(-seen_range,seen_range, facecolor='0.8', alpha=0.5)

plt.legend()

plt.ylim(-1.5, 1.5)
plt.savefig('eql perf')
plt.clf()



# cut to avoid axis issues - and see magnitude scale
interpolation_losses = np.log10(interpolation_losses)
extrapolation_losses = np.log10(extrapolation_losses)


plt.suptitle(f'log10-Loss per epoch. {eql.n_fglayers}x{eql.n_per_base} EQL')
corr = torch.corrcoef( torch.Tensor([interpolation_losses, extrapolation_losses]) )[0, 1]
best_interpol_epoch = torch.argmin(torch.Tensor(interpolation_losses))
best_extrapol_epoch = torch.argmin(torch.Tensor(extrapolation_losses))
plt.title(f'extra-intra corr={corr : .3f}; Best: inter={best_interpol_epoch}, extra={best_extrapol_epoch}')
plt.scatter(best_interpol_epoch, interpolation_losses[best_interpol_epoch], label='best interpolation', marker='x', s=50, color='k')
plt.scatter(best_extrapol_epoch, extrapolation_losses[best_extrapol_epoch], label='best extrapolation', marker='o', s=50, color='k')


plt.xlim(right=epochs, left=epochs*0.05)

plt.plot(range(len(interpolation_losses)), interpolation_losses, color='orange', label='Interpolation MSE')
plt.plot(range(len(extrapolation_losses)), extrapolation_losses, color='blue', label='Extrapolation MSE')


plt.axvspan(0, 1/4*epochs, label='No decay', alpha=0.3, color='lightgray')

plt.axvspan(1/4*epochs, 19/20*epochs, label='Weight decay', alpha=0.3, color='lightgreen')
plt.axvline(1/4*epochs, color='k')


plt.axvspan(19/20*epochs, epochs, label='L0 norm', alpha=0.3, color='lightblue')
plt.axvline(19/20*epochs, color='k')

plt.xlabel('Epoch')
plt.ylabel('log L1 Loss')
plt.legend(loc='upper right')
plt.savefig('eql training loss')