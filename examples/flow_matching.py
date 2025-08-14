import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import make_moons
from typing import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


from pyad.new_core import Tensor, MLP, Module
from pyad.optim import Adam


# ...existing code...
def odeint(f, y0, t, method='rk4', steps_per_interval=32, time_first=False, t_broadcast='match'):
    """
    Simple ODE integrator for pyad Tensors.
    - f: callable taking (y, t) by default, or (t, y) if time_first=True.
         Must return a Tensor with same shape as y.
    - y0: Tensor (initial state)
    - t: 1D iterable of time points (list/tuple/np.ndarray)
    - method: 'rk4' or 'euler'
    - steps_per_interval: sub-steps per [t[i], t[i+1]] interval
    - t_broadcast: 'scalar' -> pass scalar Tensor t; 'match' -> pass Tensor broadcastable to y's leading dims
    Returns: Tensor of shape (len(t), *y.shape)
    """
    if not isinstance(y0, Tensor):
        y = Tensor(np.array(y0))
    else:
        y = y0
    t_arr = np.asarray(t, dtype=float).ravel()
    assert t_arr.ndim == 1 and t_arr.size >= 1

    def make_t_tensor(tval, ref):
        if t_broadcast == 'match':
            lead_shape = ref.shape[:-1]  # common for (B, ..., D)
            if len(lead_shape) == 0:
                return Tensor(np.array(tval))
            return Tensor(np.full(lead_shape, tval, dtype=float))
        else:
            return Tensor(np.array(tval))

    ys = [y]
    t_prev = t_arr[0]
    for i in range(len(t_arr) - 1):
        t_next = t_arr[i + 1]
        h = (t_next - t_prev) / float(max(1, int(steps_per_interval)))
        n_sub = max(1, int(steps_per_interval))
        t_curr = t_prev
        for _ in range(n_sub):
            if method == 'euler':
                tt = make_t_tensor(t_curr, y)
                dy = f(tt, y) if time_first else f(y, tt)
                y = y + dy * h
                t_curr += h
            elif method == 'rk4':
                tt1 = make_t_tensor(t_curr, y)
                k1 = f(tt1, y) if time_first else f(y, tt1)

                tt2 = make_t_tensor(t_curr + 0.5*h, y)
                k2 = f(tt2, y + k1 * (0.5*h)) if time_first else f(y + k1 * (0.5*h), tt2)

                tt3 = make_t_tensor(t_curr + 0.5*h, y)
                k3 = f(tt3, y + k2 * (0.5*h)) if time_first else f(y + k2 * (0.5*h), tt3)

                tt4 = make_t_tensor(t_curr + h, y)
                k4 = f(tt4, y + k3 * h) if time_first else f(y + k3 * h, tt4)

                y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
                t_curr += h
            else:
                raise ValueError("method must be 'rk4' or 'euler'")
        ys.append(y)
        t_prev = t_next

    return Tensor.stack(ys, axis=0)

class VectorField(Module):
    def __init__(self, features: int, hidden_features: List[int] = [64, 64], num_freqs: int = 3):
        super().__init__()

        self.net = MLP(nin = 2 * num_freqs + features, nouts = hidden_features + [features])
        self.freqs = Tensor([i * math.pi for i in range(1, num_freqs + 1)])

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        tt = t[..., None] * self.freqs
        enc = Tensor.concatenate((tt.cos(), tt.sin()), axis=-1)
        return self.net(Tensor.concatenate((enc, x), axis=-1))

    def encode(self, x: Tensor) -> Tensor:
        # from data to noise
        return odeint(self, x, (0.0, 1.0))[-1]

    def decode(self, z: Tensor) -> Tensor:
        # from noise to data
        return odeint(self, z, (1.0, 0.0), method='rk4', steps_per_interval=64)[-1]
    
    def parameters(self):
        return [*self.net.parameters()]

class ConditionalFlowMatchingLoss(Module):
    def __init__(self, v: Module, sig_min = 1e-5):
        super().__init__()
        self.sig_min = sig_min
        self.v = v
        
    def psi_t(self, x: Tensor, t: Tensor, z: Tensor):
        # equation 22 from paper with t = 1 - t
        return (1 - t) * x + (self.sig_min + (1 - self.sig_min) * t) * z 
    
    def d_psi_t(self, x: Tensor, z:Tensor):
        # partial derivative wrt. t of equation 22
        return (1 - self.sig_min) * z - x

    def __call__(self, x: Tensor) -> Tensor:
        t = Tensor.rand(x[..., 0, None].shape)
        z = Tensor.randn(x.shape)
        y = self.psi_t(x, t, z)
        u = self.d_psi_t(x, z)
        # v is the vector field we want to learn
        vt = self.v(y, t.reshape(-1))

        return Tensor.mean((vt - u) ** 2)
    
    def parameters(self):
        return [self.v.parameters()]


v_t = VectorField(features = 2, hidden_features = [64] * 3)

# Training
CFMLoss = ConditionalFlowMatchingLoss(v_t)
optimizer = Adam(v_t.parameters(), lr=1e-4)

# get data and standardize
num_samples = 500
data, _ = make_moons(num_samples, noise=0.15)
data = StandardScaler().fit_transform(data)

# put data into dataloader
batch_size = 100
dataset = torch.from_numpy(data).float()
dataset = TensorDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

losses = []
n_epochs = 2000
for epoch in range(n_epochs):
    for batch in dataloader:
      x_0 = Tensor(batch[0].numpy())

      loss = CFMLoss(x_0)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      losses.append(loss.data)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {losses[-1]}")
        
plt.plot(losses, label="Loss")
plt.show()

# Sampling
z = Tensor.randn((100, 2))
# decode for every sample
x = Tensor([v_t.decode(z[i]).data for i in range(z.shape[0])])

plt.scatter(*x.data.T, label="Generated Samples")
#plt.scatter(*data.T, label="Original Data")
plt.legend()
plt.show()