import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
#from torch import Tensor
from typing import *
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


from pyad.new_core import Tensor, MLP

class VectorField(nn.Module):
    def __init__(self, features: int, hidden_features: List[int] = [64, 64], num_freqs: int = 3):
        super().__init__()

        self.net = MLP(nin = 2 * num_freqs + features, nouts = hidden_features + [features])
        self.freqs = Tensor([i * math.pi for i in range(1, num_freqs + 1)])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = Tensor.concatenate((t.cos(), t.sin()), axis=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(Tensor.concatenate((t, x), axis=-1))

    def encode(self, x: Tensor) -> Tensor:
        #z = torchdiffeq.odeint_adjoint(flow, x, torch.tensor((0.0, 1.0)))[-1]
        # from data to noise
        return odeint(self, x, (0.0, 1.0))[-1]

    def decode(self, z: Tensor) -> Tensor:
        #x = torchdiffeq.odeint_adjoint(flow, z, torch.tensor((1.0, 0.0)))[-1]
        # from noise to data
        return odeint(self, z, (1.0, 0.0))[-1]

class ConditionalFlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module, sig_min = 1e-4):
        super().__init__()
        self.sig_min = sig_min
        self.v = v
        
    def psi_t(self, x: Tensor, t: Tensor, z: Tensor):
        # equation 22 from paper with t = 1 - t
        return (1 - t) * x + (self.sig_min + (1 - self.sig_min) * t) * z 
    
    def d_psi_t(self, x: Tensor, z:Tensor):
        # partial derivative wrt. t of equation 22
        return (1 - self.sig_min) * z - x

    def forward(self, x: Tensor) -> Tensor:
        t = Tensor.rand(x[..., 0, None].shape)
        z = Tensor.randn(x.shape)
        y = self.psi_t(x, t, z)
        u = self.d_psi_t(x, z)
        # v is the vector field we want to learn
        vt = self.v(t.reshape(-1), y)

        return Tensor.mean((vt - u) ** 2)


v_t = VectorField(features = 2, hidden_features=[64] * 3)

# Training
CFMLoss = ConditionalFlowMatchingLoss(v_t)
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)

# get data and standardize
num_samples = 1000
data, _ = make_moons(num_samples, noise=0.15)
data = StandardScaler().fit_transform(data)

# put data into dataloader
batch_size = 100
dataset = Tensor.from_numpy(data).float()
dataset = TensorDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

losses = []
n_epochs = 1000
for epoch in range(n_epochs):
    for batch in dataloader:
      x_0 = batch[0]

      loss = CFMLoss(x_0)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      losses.append(loss.detach())
      
      
# Sampling
with torch.no_grad():
    z = Tensor.randn((10, 2))
    # decode for every sample
    x = Tensor([v_t.decode(z[i]) for i in range(z.shape[0])])

plt.hist2d(*x.T, bins=64)

plt.hist2d(*data.T, bins=64)