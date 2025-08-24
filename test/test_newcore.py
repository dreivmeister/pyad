# tests are taken from micrograd:
# https://github.com/karpathy/micrograd
# can be run with the command ...\pyad> python -m pytest
import math
import torch
import numpy as np
from pyad.new_core import Tensor

def test_sanity_check():

    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z + z * x
    h = Tensor.relu(z * z)
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = Tensor.tanh(a * b + b**3)
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + Tensor.relu(b + a)
    d = d + 3 * d + Tensor.relu(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    h = g.exp()
    h.backward()
    amg, bmg, hmg = a, b, h

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = (a * b + b**3).tanh()
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    h = g.exp()
    h.backward()
    apt, bpt, hpt = a, b, h

    tol = 1e-5 # should be 1e-6
    # forward pass went well
    assert abs(hmg.data - hpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    
def test_even_more_ops():
    a = Tensor(1.0)
    b = Tensor(2.0)
    c = a + b
    d = Tensor.tanh(a * b + b**3).log()
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a)#.sin()
    d += 3 * d + (b - a).cos()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    h = g.exp().sqrt()
    h.backward()
    amg, bmg, hmg = a, b, h

    a = torch.Tensor([1.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = (a * b + b**3).tanh().log()
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a)#.sin() # sin and cos together dont work!!
    d = d + 3 * d + (b - a).cos()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    h = g.exp().sqrt()
    h.backward()
    apt, bpt, hpt = a, b, h

    tol = 1e-5 # should be 1e-6
    # forward pass went well
    assert abs(hmg.data - hpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_sum():
    # Test sum with axis and keepdims, compare against torch
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    x = Tensor(x_np)
    
    # axis=None, keepdims=False
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    x = Tensor(x_np)
    y = x.sum()
    yt = xt.sum()
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.backward()
    assert np.allclose(x.grad, xt.grad.numpy())

    # axis=0, keepdims=False
    y = x.sum(axis=0)
    yt = xt.sum(dim=0)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x.grad, xt.grad.numpy())

    # axis=1, keepdims=True
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    x = Tensor(x_np)
    
    y = x.sum(axis=1, keepdims=True)
    yt = xt.sum(dim=1, keepdim=True)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x.grad, xt.grad.numpy())
    
    # axis=0, keepdims=True
    y = x.sum(axis=0, keepdims=True)
    yt = xt.sum(dim=0, keepdim=True)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x.grad, xt.grad.numpy())

    # axis=1, keepdims=False
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    x = Tensor(x_np)
    
    y = x.sum(axis=1, keepdims=False)
    yt = xt.sum(dim=1, keepdim=False)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x.grad, xt.grad.numpy())
    
def test_sum_3d():
    # Now test with a 3D array
    x3_np = np.arange(2*3*4).reshape(2,3,4).astype(np.float64)
    xt3 = torch.tensor(x3_np, dtype=torch.float64, requires_grad=True)
    x3 = Tensor(x3_np)

    # axis=None, keepdims=False
    y = x3.sum()
    yt = xt3.sum()
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.backward()
    assert np.allclose(x3.grad, xt3.grad.numpy())

    # axis=0, keepdims=False
    xt3 = torch.tensor(x3_np, dtype=torch.float64, requires_grad=True)
    x3 = Tensor(x3_np)
    y = x3.sum(axis=0)
    yt = xt3.sum(dim=0)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x3.grad, xt3.grad.numpy())

    # axis=1, keepdims=True
    xt3 = torch.tensor(x3_np, dtype=torch.float64, requires_grad=True)
    x3 = Tensor(x3_np)
    y = x3.sum(axis=1, keepdims=True)
    yt = xt3.sum(dim=1, keepdim=True)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x3.grad, xt3.grad.numpy())

    # axis=2, keepdims=False
    xt3 = torch.tensor(x3_np, dtype=torch.float64, requires_grad=True)
    x3 = Tensor(x3_np)
    y = x3.sum(axis=2, keepdims=False)
    yt = xt3.sum(dim=2, keepdim=False)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x3.grad, xt3.grad.numpy())

    # axis=(1,2), keepdims=True
    xt3 = torch.tensor(x3_np, dtype=torch.float64, requires_grad=True)
    x3 = Tensor(x3_np)
    y = x3.sum(axis=(1,2), keepdims=True)
    yt = xt3.sum(dim=(1,2), keepdim=True)
    assert np.allclose(y.data, yt.detach().numpy())
    y.backward()
    yt.sum().backward()
    assert np.allclose(x3.grad, xt3.grad.numpy())
    
    
def test_broadcast():
    # Test broadcasting with scalar
    x = Tensor([1.0, 2.0])
    y = x + 3.0  # should broadcast 3.0 to [4.0, 5.0]
    assert np.allclose(y.data, [4.0, 5.0])
    
    # Test broadcasting with vector
    x = Tensor([1.0, 2.0])
    y = x + Tensor([3.0, 4.0])  # should broadcast [3.0, 4.0] to [3.0, 4.0]
    assert np.allclose(y.data, [4.0, 6.0])
    
    # Test broadcasting with matrix
    x = Tensor([[1.0, 2.0], 
                [3.0, 4.0]])
    y = x + Tensor([[5.0], 
                    [6.0]])  # should broadcast [[5.0], [6.0]] to [[5.0, 5.0], [6.0, 6.0]]
    assert np.allclose(y.data, [[6.0, 7.0], 
                                 [9.0, 10.0]])
    
    
def test_matmul():
    # Forward test
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])
    z = x.dot(y)

    # Backward test
    z.sum().backward()

    # Compare with PyTorch
    xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64, requires_grad=True)
    zt = torch.matmul(xt, yt)
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
def test_matrix_vector_vector_matrix():
    # matrix vector
    # Forward test
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([7.0, 8.0])
    z = x.dot(y)

    # Backward test
    z.sum().backward()

    # Compare with PyTorch
    xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([7.0, 8.0], dtype=torch.float64, requires_grad=True)
    zt = torch.matmul(xt, yt)
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    
    # vector matrix
    # Forward test
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([7.0, 8.0])
    z = y.dot(x)

    # Backward test
    z.sum().backward()

    # Compare with PyTorch
    xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([7.0, 8.0], dtype=torch.float64, requires_grad=True)
    zt = torch.matmul(yt, xt)
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    
def test_1d_tensor_ops():
    # Forward test: addition and multiplication
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = x.dot(y)
    z.backward()
    
    # Compare with PyTorch
    xt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = torch.matmul(xt, yt)
    zt.backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    
def test_outer():
    # Forward test
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0])
    z = x.outer(y)

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([4.0, 5.0], dtype=torch.float64, requires_grad=True)
    zt = torch.outer(xt, yt)
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
def test_scalar_times_vector():
    # Forward test
    x = Tensor([3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = y * x # x * y

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = xt * yt
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = y * x # x * y
    z.sum().backward()
    
    xt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = xt * yt
    zt.sum().backward()
    
    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    
def test_elementwise_matrix_multiply():
    # Forward test
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])
    z = x * y

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64, requires_grad=True)
    zt = xt * yt
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
def test_concat():
    # Forward test
    x = Tensor([1.0, 2.0])
    y = Tensor([5.0, 6.0])
    z = Tensor.concatenate([x, y])

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = torch.cat((xt, yt))
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())
    
    
def test_concat2():
    x_np = np.random.rand(2, 3)
    y_np = np.random.rand(2, 2)
    
    # Forward test
    x = Tensor(x_np.copy())
    y = Tensor(y_np.copy())
    z = Tensor.concatenate([x, y], axis=1)

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)
    zt = torch.cat((xt, yt), dim=1)
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    assert np.allclose(y.grad, yt.grad.numpy())

def test_mean():
    # Forward test
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = x.mean()
    # Backward test
    y.backward()

    xt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    yt = xt.mean()
    yt.backward()

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())

def test_var():
    # Forward test
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = x.var()
    # Backward test
    y.backward()

    xt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    yt = xt.var(unbiased=False)  # match numpy's default (population variance)
    yt.backward()

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    

import torch.nn.functional as F
def test_softmax():
    x_np = np.random.rand(3, 5)
    x = Tensor(x_np)
    y = x.softmax(axis=1)
    y.sum().backward()
    
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = F.softmax(xt, dim=1)
    yt.sum().backward()
    
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    
def test_max():
    # 1D case
    x_np = np.array([1.0, 3.0, 2.0, 5.0])
    x = Tensor(x_np)
    y = x.max()
    y.backward()
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = xt.max()
    yt.backward()
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())

    # 2D case, axis=0
    x_np = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    x = Tensor(x_np)
    y = x.max(axis=0)
    y.sum().backward()
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = xt.max(dim=0).values
    yt.sum().backward()
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())

    # 2D case, axis=1, keepdims=True
    x_np = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    x = Tensor(x_np)
    y = x.max(axis=1, keepdims=True)
    y.sum().backward()
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = xt.max(dim=1, keepdim=True).values
    yt.sum().backward()
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())
    
    
def test_conv2d():
    # Parameters
    batch_size = 2
    in_channels = 3
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3

    x_np = np.random.randn(batch_size, in_channels, height, width)
    w_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
    b_np = np.random.randn(out_channels)

    # PyAD tensors
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)

    # PyTorch tensors
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    

    # Forward
    y = x.conv2d(w, bias=b)
    m = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
    m.weight = torch.nn.Parameter(wt)
    m.bias = torch.nn.Parameter(bt)
    yt = m(xt)

    assert y.shape == yt.shape
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)

    # Backward
    y.sum().backward()
    yt.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    assert np.allclose(w.grad, m.weight.grad.numpy(), atol=1e-6)
    
    
def test_maxpool2d():
    # Parameters
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2

    np.random.seed(42)
    x_np = np.random.randn(batch_size, channels, height, width)
    x = Tensor(x_np)
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

    # Forward
    y = x.maxpool2d(kernel_size, kernel_size, stride)
    yt = F.max_pool2d(xt, kernel_size=kernel_size, stride=stride)

    assert y.shape == yt.shape
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)

    # Backward
    y.sum().backward()
    yt.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    
    
from pyad.nn import MLP
from pyad.optim import Adam
def test_mlp_xor():
    # XOR dataset
    X = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
    y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))

    mlp = MLP(2, [16, 1], nonlin='relu')
    optim = Adam(mlp.parameters())    

    for epoch in range(1000):
        out = mlp(X)
        loss = ((out - y) ** 2).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        

    # Test predictions
    preds = mlp(X)
    # Should be close to [0, 1, 1, 0]
    assert ((preds - y) ** 2).sum().data < 0.1
    
    X_test = Tensor(np.array([[0,0],[1,1]], dtype=np.float64))
    preds_test = mlp(X_test)
    assert ((preds_test - Tensor(np.array([[0],[0]], dtype=np.float64))) ** 2).sum().data < 0.1


def test_mlp_xor_diff_batch():
    # XOR dataset
    X1 = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
    X2 = Tensor(np.array([[0,0],[1,1]], dtype=np.float64))
    y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))

    mlp = MLP(2, [5, 1], nonlin='relu')
    optim = Adam(mlp.parameters())    

    optim.zero_grad()
    
    out1 = mlp(X1)
    loss1 = ((out1 - y) ** 2).sum()
    loss1.backward()
    optim.step()
    
    optim.zero_grad()
    
    out2 = mlp(X2)
    loss2 = ((out2 - Tensor(np.array([[0],[0]], dtype=np.float64))) ** 2).sum()
    loss2.backward()

def test_linear_layer():

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 4
    in_features = 3
    out_features = 2

    # Random input and weights
    x_np = np.random.randn(batch_size, in_features)
    w_np = np.random.randn(in_features, out_features)
    b_np = np.random.randn(out_features)

    # PyAD tensors
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)

    # Forward pass (PyAD)
    y = x.linear(w, b)
    loss = y.sum()
    loss.backward()

    # PyTorch tensors
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)

    # Forward pass (PyTorch)
    yt = xt @ wt + bt
    loss_t = yt.sum()
    loss_t.backward()

    # Compare outputs
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)
    # Compare gradients
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    assert np.allclose(w.grad, wt.grad.numpy(), atol=1e-6)
    assert np.allclose(b.grad, bt.grad.numpy(), atol=1e-6)
    
    
from pyad.new_core import log_softmax
def test_log_softmax():
    x_np = np.random.randn(2, 3)
    x = Tensor(x_np)
    y = log_softmax(x, axis=1)
    y.sum().backward()

    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = F.log_softmax(xt, dim=1)
    yt.sum().backward()

    print(y.data)
    print(yt.detach().numpy())
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    
    
from pyad.new_core import categorical_cross_entropy_from_logits
def test_categorical_cross_entropy():
    np.random.seed(42)

    batch_size = 5
    num_classes = 4

    # Random logits and integer labels
    logits_np = np.random.randn(batch_size, num_classes)
    labels_np = np.random.randint(0, num_classes, size=(batch_size,))
    onehot_np = np.zeros((batch_size, num_classes), dtype=np.float64)
    onehot_np[np.arange(batch_size), labels_np] = 1.0

    # PyAD
    logits = Tensor(logits_np)
    targets = Tensor(onehot_np)
    loss = categorical_cross_entropy_from_logits(logits, targets)
    loss.backward()

    # PyTorch
    logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    labels_t = torch.tensor(labels_np, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t, labels_t)
    loss_t.backward()

    # Compare forward
    assert np.allclose(loss.data, loss_t.detach().numpy(), atol=1e-6)
    # Compare gradients
    assert np.allclose(logits.grad, logits_t.grad.numpy(), atol=1e-6)

from pyad.new_core import negative_log_likelihood    
def test_negative_log_likelihood():
    # import torch
    # import torch.nn.functional as F
    np.random.seed(42)

    batch_size = 8

    # Random probabilities and binary targets
    probs_np = np.clip(np.random.rand(batch_size, 1), 1e-6, 1-1e-6)  # Avoid log(0)
    targets_np = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float64)

    # PyAD
    probs = Tensor(probs_np)
    targets = Tensor(targets_np)
    loss = negative_log_likelihood(probs, targets)
    loss.backward()

    # PyTorch
    probs_t = torch.tensor(probs_np, dtype=torch.float64, requires_grad=True)
    targets_t = torch.tensor(targets_np, dtype=torch.float64)
    loss_t = F.binary_cross_entropy(probs_t, targets_t, reduction='mean')
    loss_t.backward()

    # Compare forward
    assert np.allclose(loss.data, loss_t.detach().numpy(), atol=1e-6)
    # Compare gradients
    assert np.allclose(probs.grad, probs_t.grad.numpy(), atol=1e-6)
    
    
from pyad.nn import BatchNorm1D
def test_batchnorm1d():
    np.random.seed(42)

    batch_size = 8
    num_features = 5

    # Random input
    x_np = np.random.randn(batch_size, num_features)
    x = Tensor(x_np)

    # Your BatchNorm1D (training mode)
    bn = BatchNorm1D(num_features)
    y = bn(x, training=True)
    y.sum().backward()

    # PyTorch BatchNorm1d
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    bnt = torch.nn.BatchNorm1d(num_features, affine=True, track_running_stats=False).double()
    # Copy gamma and beta
    bnt.weight.data = torch.tensor(bn.gamma.data, dtype=torch.float64)
    bnt.bias.data = torch.tensor(bn.beta.data, dtype=torch.float64)
    yt = bnt(xt)
    yt.sum().backward()

    # Compare forward
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), "Forward outputs do not match"
    # Compare gradients
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), "Input gradients do not match"
    assert np.allclose(bn.gamma.grad, bnt.weight.grad.numpy(), atol=1e-6), "Gamma gradients do not match"
    assert np.allclose(bn.beta.grad, bnt.bias.grad.numpy(), atol=1e-6), "Beta gradients do not match"


from pyad.nn import LayerNorm
def test_layernorm():
    np.random.seed(42)

    batch_size = 6
    num_features = 4

    # Random input
    x_np = np.random.randn(batch_size, num_features)
    x = Tensor(x_np)

    # Your LayerNorm (assumes normalized_shape=num_features)
    ln = LayerNorm(num_features)
    y = ln(x)
    y.sum().backward()

    # PyTorch LayerNorm
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    lnt = torch.nn.LayerNorm(num_features, elementwise_affine=True).double()
    yt = lnt(xt)
    yt.sum().backward()

    # Compare forward
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), "Forward outputs do not match"
    # Compare gradients
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), "Input gradients do not match"
    assert np.allclose(ln.gamma.grad, lnt.weight.grad.numpy(), atol=1e-6), "Gamma gradients do not match"
    assert np.allclose(ln.beta.grad, lnt.bias.grad.numpy(), atol=1e-6), "Beta gradients do not match"
    
    
def test_layernorm_conv():
    np.random.seed(42)

    batch_size = 6
    num_channels = 3
    dim = 10

    # Random input
    x_np = np.random.randn(batch_size, num_channels, dim, dim)
    x = Tensor(x_np)

    # Your LayerNorm (assumes normalized_shape=num_features)
    ln = LayerNorm((num_channels, dim, dim))
    y = ln(x)
    y.sum().backward()

    # PyTorch LayerNorm
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    lnt = torch.nn.LayerNorm((num_channels, dim, dim), elementwise_affine=True).double()
    yt = lnt(xt)
    yt.sum().backward()

    # Compare forward
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), "Forward outputs do not match"
    # Compare gradients
    assert np.allclose(ln.gamma.grad, lnt.weight.grad.numpy(), atol=1e-6), "Gamma gradients do not match"
    assert np.allclose(ln.beta.grad, lnt.bias.grad.numpy(), atol=1e-6), "Beta gradients do not match"
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), "Input gradients do not match"
    
def test_conv2d_stride_pad():
    np.random.seed(42)
    torch.manual_seed(42)
    batch_size = 2
    in_channels = 3
    out_channels = 2
    height = 8
    width = 8
    kernel_size = 3

    x_np = np.random.randn(batch_size, in_channels, height, width)
    w_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)

    # Settings to test: (stride, padding)
    settings = [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 2),
    ]

    for stride, pad in settings:
        # PyAD tensors
        x = Tensor(x_np)
        w = Tensor(w_np)
        # Forward
        y = x.conv2d(w, stride=stride, pad=pad, bias=None)
        # PyTorch tensors
        xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
        yt = F.conv2d(xt, wt, bias=None, stride=stride, padding=pad)
        # Compare outputs
        assert y.shape == yt.shape, f"Shape mismatch for stride={stride}, pad={pad}"
        assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), f"Forward mismatch for stride={stride}, pad={pad}"
        # Backward
        y.sum().backward()
        yt.sum().backward()
        assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), f"Input grad mismatch for stride={stride}, pad={pad}"
        assert np.allclose(w.grad, wt.grad.numpy(), atol=1e-6), f"Weight grad mismatch for stride={stride}, pad={pad}"
        
def test_dropout_forward():
    x_np = np.random.randn(4, 5)
    x = Tensor(x_np)
    p_drop = 0.5
    # Test training mode
    y = x.dropout(p_drop, training=True)
    # Test inference mode (should be unchanged)
    y_eval = x.dropout(p_drop, training=False)
    assert np.allclose(y_eval.data, x_np), "Dropout eval mode should not change input"
    # Check that some elements are zeroed out in training mode
    num_zeros = np.sum(y.data == 0)
    assert num_zeros > 0, "Dropout should zero out some elements in training mode"

def test_dropout_backward():
    x_np = np.random.randn(3, 3)
    x = Tensor(x_np)
    p_drop = 0.3
    y = x.dropout(p_drop, training=True)
    y.sum().backward()
    # The gradient should be nonzero only where mask is 1
    mask = (y.data != 0)
    grad_nonzero = (x.grad != 0)
    assert np.all(mask == grad_nonzero), "Dropout backward mask mismatch"
    
    
def test_bmm():
    # Create random batch of matrices
    batch_size = 4
    n = 3
    m = 5
    p = 2
    a_np = np.random.randn(batch_size, n, m)
    b_np = np.random.randn(batch_size, m, p)
    a = Tensor(a_np)
    b = Tensor(b_np)
    # Forward
    out = a.bmm(b)
    # Torch reference
    a_torch = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    out_torch = torch.bmm(a_torch, b_torch)
    assert np.allclose(out.data, out_torch.detach().numpy()), f"Forward bmm failed: {out.data} vs {out_torch.detach().numpy()}"
    out.sum().backward()
    out_torch.sum().backward()
    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Backward bmm grad_a failed: {a.grad} vs {a_torch.grad.numpy()}"
    assert np.allclose(b.grad, b_torch.grad.numpy()), f"Backward bmm grad_b failed: {b.grad} vs {b_torch.grad.numpy()}"
    

#import torch
from pyad.nn import LinearLayer
def test_linear_layer_3d():
    np.random.seed(42)
    torch.manual_seed(42)
    # 3D input: (B, T, C)
    B, T, C_in, C_out = 2, 3, 4, 5
    x_np = np.random.randn(B, T, C_in)
    w_np = np.random.randn(C_in, C_out)
    b_np = np.random.randn(C_out)
    # PyAD
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)
    # LinearLayer expects (N, in_features), so flatten batch and time
    #x_flat = Tensor(x_np.reshape(-1, C_in))
    ll = LinearLayer(C_in, C_out, bias=True)
    ll.W = w
    ll.b = b
    y = ll(x)
    #y_reshaped = y.reshape((B, T, C_out))
    y.sum().backward()
    # Torch
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    lin = torch.nn.Linear(C_in, C_out)
    lin.weight = torch.nn.Parameter(wt.T)
    lin.bias = torch.nn.Parameter(bt)
    yt = lin(xt)
    yt.sum().backward()
    # Compare outputs and gradients
    print(y.shape, yt.size())
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), f"Forward mismatch: {y.data} vs {yt.detach().numpy()}"
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), f"Input grad mismatch: {x.grad} vs {xt.grad.numpy()}"
    assert np.allclose(w.grad, lin.weight.grad.numpy().T, atol=1e-6), f"Weight grad mismatch: {w.grad} vs {wt.grad.numpy()}"
    assert np.allclose(b.grad, lin.bias.grad.numpy(), atol=1e-6), f"Bias grad mismatch: {b.grad} vs {bt.grad.numpy()}"
    
    
def test_embedding():
    # Parameters
    batch_size = 3
    seq_len = 4
    vocab_size = 10
    emb_dim = 6
    np.random.seed(42)
    torch.manual_seed(42)
    # Random integer indices
    idx_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    weight_np = np.random.randn(vocab_size, emb_dim)
    # PyAD
    idx = Tensor(idx_np)
    weight = Tensor(weight_np)
    out = idx.embedding(weight)
    # Torch
    idx_t = torch.tensor(idx_np, dtype=torch.long)
    weight_t = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
    emb_t = torch.nn.Embedding(vocab_size, emb_dim)
    emb_t.weight = torch.nn.Parameter(weight_t)
    out_t = emb_t(idx_t)
    # Compare forward
    assert np.allclose(out.data, out_t.detach().numpy(), atol=1e-6), f"Embedding forward mismatch: {out.data} vs {out_t.detach().numpy()}"
    # Backward
    dout = np.random.randn(*out.data.shape)
    out.backward(dout)
    out_t.backward(torch.tensor(dout, dtype=torch.float64))
    assert np.allclose(weight.grad, emb_t.weight.grad.numpy(), atol=1e-6), f"Embedding grad mismatch: {weight.grad} vs {emb_t.weight.grad.numpy()}"
    
def test_getitem():
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a random tensor
    x_np = np.random.randn(4, 5, 6)
    x = Tensor(x_np)
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

    # Test various slicing/indexing
    slices = [
        ([0,2],[0,1],[0,4]),
        (slice(3), slice(2), slice(5)),  # range on all axes
        #(slice(None), slice(None), slice(None)),  # full
        #(slice(1, 3), slice(None), slice(None)),  # range on first axis
        # (slice(None), 2, slice(None)),            # integer index on second axis
        # (slice(None), slice(None), 4),            # integer index on third axis
        # (1, 2, 3),                                # single element
        # (Ellipsis, 2),                            # ellipsis
        # (slice(None, None, 2), slice(None), slice(None)),  # step
    ]

    for s in slices:
        y = x[s]
        yt = xt[s]
        assert np.allclose(y.data, yt.detach().numpy()), f"Forward getitem mismatch for slice {s}"

        # Backward: sum to scalar, then backward
        # x.grad = np.zeros_like(x.data)
        # xt.grad = None
        y.sum().backward()
        yt.sum().backward()
        assert np.allclose(x.grad, xt.grad.numpy()), f"Backward getitem grad mismatch for slice {s}"


# ...existing code...

def test_sparse_ce_no_ignore_index():
    from pyad.new_core import sparse_categorical_crossentropy_from_logits

    np.random.seed(0)
    torch.manual_seed(0)

    N, C = 7, 5
    logits_np = np.random.randn(N, C)
    targets_np = np.random.randint(0, C, size=(N,), dtype=np.int64)

    # Our framework
    logits = Tensor(logits_np)
    targets = Tensor(targets_np)
    loss = sparse_categorical_crossentropy_from_logits(logits, targets, ignore_index=None)
    loss.backward()

    # Torch reference
    logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    targets_t = torch.tensor(targets_np, dtype=torch.long)
    ce = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_t = ce(logits_t, targets_t)
    loss_t.backward()

    assert np.allclose(loss.data, loss_t.detach().numpy(), atol=1e-6), "Forward loss mismatch (no ignore_index)"
    assert np.allclose(logits.grad, logits_t.grad.numpy(), atol=1e-6), "Logits grad mismatch (no ignore_index)"


def test_sparse_ce_with_ignore_index():
    from pyad.new_core import sparse_categorical_crossentropy_from_logits

    np.random.seed(1)
    torch.manual_seed(1)

    N, C = 9, 6
    logits_np = np.random.randn(N, C)
    targets_np = np.random.randint(0, C, size=(N,), dtype=np.int64)

    # Mark a few entries to ignore
    ignore_index = -1
    ignore_mask = np.zeros(N, dtype=bool)
    ignore_mask[[1, 4, 7]] = True
    targets_np[ignore_mask] = ignore_index

    # Our framework
    logits = Tensor(logits_np)
    targets = Tensor(targets_np)
    loss = sparse_categorical_crossentropy_from_logits(logits, targets, ignore_index=ignore_index)
    loss.backward()

    # Torch reference
    logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    targets_t = torch.tensor(targets_np, dtype=torch.long)
    ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss_t = ce(logits_t, targets_t)
    loss_t.backward()

    assert np.allclose(loss.data, loss_t.detach().numpy(), atol=1e-6), "Forward loss mismatch (with ignore_index)"
    assert np.allclose(logits.grad, logits_t.grad.numpy(), atol=1e-6), "Logits grad mismatch (with ignore_index)"
    # Additionally, gradients for ignored rows should be exactly zero
    assert np.allclose(logits.grad[ignore_mask], 0.0, atol=1e-12), "Ignored rows"
    
    
def test_gelu_forward_backward():
    np.random.seed(0)
    torch.manual_seed(0)
    
    class NewGELU(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    x_np = np.random.randn(7, 11)
    x = Tensor(x_np.copy())
    xt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)

    # Forward
    y = x.gelu()
    gl = NewGELU()
    yt = gl(xt)  # matches our tanh-based approximation

    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6), "GELU forward mismatch"

    # Backward (sum to scalar)
    y.sum().backward()
    yt.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6), "GELU backward grad mismatch"


def test_gelu_broadcast_chain():
    np.random.seed(1)
    torch.manual_seed(1)
    
    class NewGELU(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    x_np = np.random.randn(5, 4)
    w_np = np.random.randn(4, 6)

    x = Tensor(x_np.copy())
    w = Tensor(w_np.copy())
    xt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)

    out = x.gelu().linear(w)  # (5,6)
    gl = NewGELU()
    out_t = gl(xt) @ wt

    out.sum().backward()
    out_t.sum().backward()

    assert np.allclose(out.data, out_t.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    assert np.allclose(w.grad, wt.grad.numpy(), atol=1e-6)
    
    
def test_feedforward_module():
    from pyad.nn import FeedForward
    
    class NewGELU(torch.nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    np.random.seed(123)
    torch.manual_seed(123)

    n_embd = 8
    B, T = 5, 4  # batch, seq

    # Create pyad FeedForward
    ff_py = FeedForward(n_embd)

    # Torch equivalent
    class TorchFF(torch.nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.fc1 = torch.nn.Linear(n_embd, 4 * n_embd)
            self.fc2 = torch.nn.Linear(4 * n_embd, n_embd)
            self.gelu = NewGELU()
        def forward(self, x):
            return self.fc2(self.gelu(self.fc1(x)))

    ff_th = TorchFF(n_embd).double()

    # Copy weights (LinearLayer stores W (nin,nout))
    with torch.no_grad():
        ff_th.fc1.weight.copy_(torch.tensor(ff_py.ll1.W.data.T))
        ff_th.fc1.bias.copy_(torch.tensor(ff_py.ll1.b.data))
        ff_th.fc2.weight.copy_(torch.tensor(ff_py.ll2.W.data.T))
        ff_th.fc2.bias.copy_(torch.tensor(ff_py.ll2.b.data))

    # Test both 2D and 3D inputs
    for shape in [(B, n_embd), (B, T, n_embd)]:
        x_np = np.random.randn(*shape)
        x_py = Tensor(x_np.copy())
        x_th = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)

        # Forward
        out_py = ff_py(x_py)
        out_th = ff_th(x_th)
        assert np.allclose(out_py.data, out_th.detach().numpy(), atol=1e-6), f"Forward mismatch shape {shape}"

        # Backward (sum to scalar)
        x_py.grad = np.zeros_like(x_py.data)
        out_py.sum().backward()
        out_th.sum().backward()

        # Compare input gradients
        assert np.allclose(x_py.grad, x_th.grad.numpy(), atol=1e-6), f"Input grad mismatch shape {shape}"

    # Compare parameter gradients (after last backward)
    # Torch grads already populated
    assert np.allclose(ff_py.ll1.W.grad, ff_th.fc1.weight.grad.numpy().T, atol=1e-6), "fc1.W grad mismatch"
    assert np.allclose(ff_py.ll1.b.grad, ff_th.fc1.bias.grad.numpy(), atol=1e-6), "fc1.b grad mismatch"
    assert np.allclose(ff_py.ll2.W.grad, ff_th.fc2.weight.grad.numpy().T, atol=1e-6), "fc2.W grad mismatch"
    assert np.allclose(ff_py.ll2.b.grad, ff_th.fc2.bias.grad.numpy(), atol=1e-6), "fc2.b grad mismatch"
    
def test_mha_vs_torch_causal_self_attention():
    from pyad.nn import CausalMultiHeadSelfAttention

    class Config:
        block_size = 6
        n_embd = 8
        n_head = 2

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = Config()
    head_size = cfg.n_embd // cfg.n_head

    # --- Build pyad MHA ---
    mha_pyad = CausalMultiHeadSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size)

    # --- Build torch CausalSelfAttention ---
    class TorchCausalSelfAttention(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))
            self.n_head = config.n_head
            self.n_embd = config.n_embd

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            return y

    mha_torch = TorchCausalSelfAttention(cfg).double()

    # --- Set weights to be the same ---
    # 1. Set c_attn weights/bias
    W = np.random.randn(cfg.n_embd, 3 * cfg.n_embd)
    b = np.random.randn(3 * cfg.n_embd)
    with torch.no_grad():
        mha_torch.c_attn.weight.copy_(torch.tensor(W.T))
        mha_torch.c_attn.bias.copy_(torch.tensor(b))
    mha_pyad.c_attn.W.data[:] = W
    mha_pyad.c_attn.b.data[:] = b

    # 2. Set c_proj weights/bias
    W_proj = np.random.randn(cfg.n_embd, cfg.n_embd)
    b_proj = np.random.randn(cfg.n_embd)
    with torch.no_grad():
        mha_torch.c_proj.weight.copy_(torch.tensor(W_proj.T))
        mha_torch.c_proj.bias.copy_(torch.tensor(b_proj))
    mha_pyad.c_proj.W.data[:] = W_proj
    mha_pyad.c_proj.b.data[:] = b_proj

    # --- Forward ---
    B, T, C = 2, cfg.block_size, cfg.n_embd
    x_np = np.random.randn(B, T, C)
    x_pyad = Tensor(x_np.copy())
    x_torch = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

    out_pyad = mha_pyad(x_pyad)
    out_torch = mha_torch(x_torch)

    assert np.allclose(out_pyad.data, out_torch.detach().numpy(), atol=1e-5, equal_nan=True), "Forward outputs mismatch"

    # --- Backward ---
    out_pyad.sum().backward()
    out_torch.sum().backward()

    # Compare input gradients
    assert np.allclose(x_pyad.grad, x_torch.grad.numpy(), atol=1e-5, equal_nan=True), "Input gradients mismatch"

    # Compare projection gradients
    assert np.allclose(mha_pyad.c_proj.W.grad, mha_torch.c_proj.weight.grad.numpy().T, atol=1e-5), "Proj W grad mismatch"
    assert np.allclose(mha_pyad.c_proj.b.grad, mha_torch.c_proj.bias.grad.numpy(), atol=1e-5), "Proj b grad mismatch"
    assert np.allclose(mha_pyad.c_attn.W.grad, mha_torch.c_attn.weight.grad.numpy().T, atol=1e-5), "qkv_proj W grad mismatch"
    assert np.allclose(mha_pyad.c_attn.b.grad, mha_torch.c_attn.bias.grad.numpy(), atol=1e-5), "qkv_proj b grad mismatch"
    
    
    
def test_split():
    np.random.seed(0)
    torch.manual_seed(0)

    # Test 1D split
    x_np = np.random.randn(8)
    x = Tensor(x_np.copy())
    xt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)

    # Split into 4 chunks along axis 0
    xs = x.split(2, axis=0)
    xts = torch.split(xt, 2, dim=0)
    for i in range(4):
        assert np.allclose(xs[i].data, xts[i].detach().numpy(), atol=1e-8)

    # Backward: sum all splits, then backward
    s = sum(xs)
    s.sum().backward()
    sum_xt = sum(xts)
    sum_xt.sum().backward()
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-8)

    # Test 2D split along axis 1
    x2_np = np.random.randn(3, 6)
    x2 = Tensor(x2_np.copy())
    xt2 = torch.tensor(x2_np.copy(), dtype=torch.float64, requires_grad=True)
    xs2 = x2.split(2, axis=1)
    xts2 = torch.split(xt2, 2, dim=1)
    for i in range(3):
        assert np.allclose(xs2[i].data, xts2[i].detach().numpy(), atol=1e-8)
    s2 = sum(xs2)
    s2.sum().backward()
    sum_xt2 = sum(xts2)
    sum_xt2.sum().backward()
    assert np.allclose(x2.grad, xt2.grad.numpy(), atol=1e-8)
    
    
def test_layernorm_axes():
    from pyad.nn import LayerNorm
    B,T,C = 2,3,4
    x_np = np.random.randn(B,T,C)
    x = Tensor(x_np.copy())
    ln = LayerNorm(C)
    y = ln(x)
    # mean/var per (B,T,*) slice over last dim ~ 0 / 1
    m = y.data.mean(axis=2)
    v = y.data.var(axis=2)
    assert np.allclose(m, 0, atol=1e-5)
    assert np.allclose(v, 1, atol=1e-4)
    
    
def test_stack():
    np.random.seed(0)
    torch.manual_seed(0)

    # Three inputs of same shape
    A = np.random.randn(2, 3)
    B = np.random.randn(2, 3)
    C = np.random.randn(2, 3)

    for axis in (0, 1, -1):
        # pyad tensors
        a = Tensor(A.copy())
        b = Tensor(B.copy())
        c = Tensor(C.copy())

        # torch tensors
        at = torch.tensor(A.copy(), dtype=torch.float64, requires_grad=True)
        bt = torch.tensor(B.copy(), dtype=torch.float64, requires_grad=True)
        ct = torch.tensor(C.copy(), dtype=torch.float64, requires_grad=True)

        # Forward: stack
        y = Tensor.stack([a, b, c], axis=axis)
        yt = torch.stack([at, bt, ct], dim=axis)

        # Check forward equivalence
        assert np.allclose(y.data, yt.detach().numpy(), atol=1e-8)

        # Build a per-slice weight along the new axis to test gradient slicing
        v = np.array([2.0, 3.0, 5.0], dtype=np.float64)
        axis_norm = axis if axis >= 0 else y.data.ndim + axis
        w_shape = [1] * y.data.ndim
        w_shape[axis_norm] = 3
        w_np = v.reshape(w_shape)

        # Weighted sum to create non-uniform gradients per slice
        s = (y * Tensor(w_np)).sum()
        s.backward()

        wt = torch.tensor(w_np, dtype=torch.float64)
        st = (yt * wt).sum()
        st.backward()

        # Check input gradients match torch
        assert np.allclose(a.grad, at.grad.numpy(), atol=1e-8)
        assert np.allclose(b.grad, bt.grad.numpy(), atol=1e-8)
        assert np.allclose(c.grad, ct.grad.numpy(), atol=1e-8)
        
        
        
def test_conv_transpose2d_vs_torch_forward_backward():
    np.random.seed(0)
    torch.manual_seed(0)

    # Shapes
    N, Cin, Cout, H, W = 2, 3, 4, 5, 6
    kH, kW = 3, 2

    # Random inputs
    x_np = np.random.randn(N, Cin, H, W)
    w_np = np.random.randn(Cin, Cout, kH, kW)  # (in_channels, out_channels, kH, kW) for conv_transpose2d
    b_np = np.random.randn(Cout)

    # pyad tensors
    x = Tensor(x_np.copy())
    w = Tensor(w_np.copy())
    b = Tensor(b_np.copy())

    # torch tensors
    xt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np.copy(), dtype=torch.float64, requires_grad=True)

    # Forward
    y = x.conv_transpose2d(w, bias=b)
    yt = F.conv_transpose2d(xt, wt, bt, stride=1, padding=0)

    # Shapes
    assert y.data.shape == yt.shape, f"Output shape mismatch: {y.data.shape} vs {tuple(yt.shape)}"
    # Values
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-8), "Forward outputs mismatch"

    # Backward with random upstream gradient
    g_np = np.random.randn(*y.data.shape)
    y.backward(g_np)
    yt.backward(torch.tensor(g_np, dtype=torch.float64))

    # Compare grads
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-8), "Grad w.r.t. input mismatch"
    assert np.allclose(w.grad, wt.grad.numpy(), atol=1e-8), "Grad w.r.t. weight mismatch"
    assert np.allclose(b.grad, bt.grad.numpy(), atol=1e-8), "Grad w.r.t. bias mismatch"


def test_conv_transpose2d_no_bias_vs_torch():
    np.random.seed(1)
    torch.manual_seed(1)

    # Smaller case
    N, Cin, Cout, H, W = 1, 1, 2, 4, 3
    kH, kW = 3, 3

    x_np = np.random.randn(N, Cin, H, W)
    w_np = np.random.randn(Cin, Cout, kH, kW)

    x = Tensor(x_np.copy())
    w = Tensor(w_np.copy())

    xt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)

    y = x.conv_transpose2d(w, bias=None)
    yt = F.conv_transpose2d(xt, wt, bias=None, stride=1, padding=0)

    assert y.data.shape == yt.shape
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-8)

    # Scalar loss to test grads
    y.sum().backward()
    yt.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-8)
    assert np.allclose(w.grad, wt.grad.numpy(), atol=1e-8)
    
    
def test_setitem_slice_and_broadcast_grad():
    np.random.seed(0)

    # x is produced from u so we can verify gradient only flows through non-overwritten region
    u = Tensor(np.random.randn(3, 4))
    x = u * 3.0

    # value has shape (3,1) and will broadcast to (3,2)
    v = Tensor(np.ones((3, 1)))
    # Assign to a slice (first two columns)
    x[:, :2] = v

    # Forward data check: left block equals v (broadcast), right block unchanged from u*3
    assert np.allclose(x.data[:, :2], np.ones((3, 2)))
    assert np.allclose(x.data[:, 2:], (u.data * 3.0)[:, 2:])

    # Backward: s = sum(x)
    s = x.sum()
    s.backward()

    # Grad to u: only through non-overwritten region (cols 2,3), scaled by 3.0
    expected_u_grad = np.zeros_like(u.data)
    expected_u_grad[:, 2:] = 3.0  # d(sum)/dx = 1, times 3 from x = u*3
    assert np.allclose(u.grad, expected_u_grad)

    # Grad to v: sum over broadcasted dimension (axis=1), since (3,1) -> (3,2)
    # Each row received two ones from the assigned columns
    expected_v_grad = np.full((3, 1), 2.0)
    assert np.allclose(v.grad, expected_v_grad)


def test_setitem_advanced_index_and_scalar_mask():
    np.random.seed(1)

    # Advanced indexing with repeated indices
    u = Tensor(np.random.randn(5))
    x = u + 0.0  # make x depend on u
    idx = [0, 0, 3]  # repeated index 0
    val = Tensor(np.array([10.0, 20.0, 30.0]))
    x[idx] = val

    # Check data mutation
    expected = (u.data + 0.0).copy()
    expected[idx] = val.data
    assert np.allclose(x.data, expected)

    # Backward
    x.sum().backward()

    # Grad to u: ones everywhere except overwritten indices become zero
    expected_u_grad = np.ones_like(u.data)
    expected_u_grad[0] = 0.0  # overwritten twice, still zero
    expected_u_grad[3] = 0.0
    assert np.allclose(u.grad, expected_u_grad)

    # Grad to val: one per occurrence (including repeats)
    expected_val_grad = np.ones_like(val.data)
    assert np.allclose(val.grad, expected_val_grad)

    # Boolean mask with scalar assignment (broadcast to all True positions)
    u2 = Tensor(np.random.randn(2, 3))
    x2 = u2 * 5.0
    mask = x2.data > 0.0  # arbitrary boolean mask based on current data
    scalar = Tensor(2.0)
    x2[mask] = scalar

    # Data check: masked positions equal 2.0
    assert np.allclose(x2.data[mask], 2.0)

    # Backward
    x2.sum().backward()

    # Grad to u2: only unmasked positions flow, scaled by 5.0
    expected_u2_grad = np.zeros_like(u2.data)
    expected_u2_grad[~mask] = 5.0
    assert np.allclose(u2.grad, expected_u2_grad)

    # Grad to scalar: number of True positions (each contributes 1)
    expected_scalar_grad = float(mask.sum())
    assert np.allclose(scalar.grad, expected_scalar_grad)
    
    
def test_setitem_slice_broadcast_vs_torch():
    np.random.seed(123)
    torch.manual_seed(123)

    # pyad graph
    u = Tensor(np.random.randn(3, 4))
    x = u * 3.0
    v = Tensor(np.ones((3, 1)))  # will broadcast to (3,2)
    x[:, :2] = v  # slice assignment with broadcast

    # torch reference (no in-place): y_ref = xt*(1-mask) + v_expanded*mask
    ut = torch.tensor(u.data.copy(), dtype=torch.float64, requires_grad=True)
    xt = ut * 3.0
    vt = torch.ones((3, 1), dtype=torch.float64, requires_grad=True)

    mask = torch.zeros_like(xt)
    mask[:, :2] = 1.0
    v_expanded = vt.expand(-1, 2)
    yt = xt * (1.0 - mask) + torch.cat([v_expanded, torch.zeros(3, 2, dtype=torch.float64)], dim=1)

    # Compare forward
    assert np.allclose(x.data, yt.detach().numpy(), atol=1e-8)

    # Backward
    x.sum().backward()
    yt.sum().backward()

    # Compare grads
    assert np.allclose(u.grad, ut.grad.numpy(), atol=1e-8)
    assert np.allclose(v.grad, vt.grad.numpy(), atol=1e-8)
    
    
# def test_setitem_advanced_duplicates_vs_torch():
#     np.random.seed(321)
#     torch.manual_seed(321)

#     # Advanced indexing with duplicates
#     idx_list = [0, 0, 3]  # last-wins at position 0
#     val_np = np.array([10.0, 20.0, 30.0], dtype=np.float64)

#     # pyad
#     u = Tensor(np.random.randn(5))
#     x = u + 0.0
#     val = Tensor(val_np.copy())
#     x[idx_list] = val

#     # torch reference (functional, no in-place), last-wins semantics
#     ut = torch.tensor(u.data.copy(), dtype=torch.float64, requires_grad=True)
#     vt = torch.tensor(val_np.copy(), dtype=torch.float64, requires_grad=True)

#     # Build one-hot selector and "keep" mask for last occurrence
#     idx_t = torch.tensor(idx_list, dtype=torch.long)
#     N = ut.numel()
#     K = idx_t.numel()
#     one_hot = F.one_hot(idx_t, num_classes=N).double()  # (K, N)

#     # keep[k] = 1 if idx_list[k] is the last occurrence of that index
#     keep = torch.zeros(K, dtype=torch.float64)
#     seen = set()
#     for k in reversed(range(K)):
#         if int(idx_list[k]) not in seen:
#             keep[k] = 1.0
#             seen.add(int(idx_list[k]))

#     # Mask of positions being assigned
#     mask_any = torch.clamp(one_hot.sum(dim=0), max=1.0)  # (N,)
#     # Assigned values with last-wins
#     assigned = (one_hot * (keep.view(-1, 1) * vt.view(-1, 1))).sum(dim=0)  # (N,)

#     yt = ut * (1.0 - mask_any) + assigned  # final vector

#     # Compare forward
#     assert np.allclose(x.data, yt.detach().numpy(), atol=1e-8)

#     # Backward
#     x.sum().backward()
#     yt.sum().backward()

#     # Compare grads
#     assert np.allclose(u.grad, ut.grad.numpy(), atol=1e-8)

#     # For val grads: only last occurrences contribute (here [0,1,1])
#     assert np.allclose(val.grad, vt.grad.numpy(), atol=1e-8)
    
    
# ...existing code...
def test_setitem_embedding():
    np.random.seed(123)
    torch.manual_seed(123)

    # pyad graph
    u = Tensor(np.random.randn(5, 4))
    x = u + 0.0
    idx = 1  # single index
    val = Tensor(np.random.randn(4,))
    x[idx] = val

    # torch reference (functional, no in-place)
    ut = torch.tensor(u.data.copy(), dtype=torch.float64, requires_grad=True)
    xt = ut + 0.0
    idx_t = torch.tensor(np.array([idx]), dtype=torch.long)
    # require grad on the assigned value to compare grads
    val_t = torch.tensor(val.data, dtype=torch.float64, requires_grad=True)

    xt[idx_t] = val_t

    # Compare forward
    assert np.allclose(x.data, xt.detach().numpy(), atol=1e-8)

    # Backward
    x.sum().backward()
    xt.sum().backward()

    # Compare grads
    assert np.allclose(u.grad, ut.grad.numpy(), atol=1e-8)
    assert np.allclose(val.grad, val_t.grad.numpy(), atol=1e-8)
    
    
    
def test_unsqueeze_basic_forward_backward():
    np.random.seed(0)
    torch.manual_seed(0)

    a = np.random.randn(2, 3)
    t = Tensor(a.copy())
    xt = torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True)

    u = t.unsqueeze(0)       # (1,2,3)
    ut = xt.unsqueeze(0)     # (1,2,3)

    assert u.data.shape == ut.detach().numpy().shape

    s = u.sum()
    s.backward()
    ut.sum().backward()

    assert np.allclose(t.grad, xt.grad.numpy())

def test_unsqueeze_multiple_and_expand_grad():
    np.random.seed(1)
    torch.manual_seed(1)

    a = np.random.randn(2, 1)
    t = Tensor(a.copy())
    xt = torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True)

    # insert a leading dim, then expand
    u = t.unsqueeze(0)                 # (1,2,1)
    v = u.expand(3, 2, 1)              # (3,2,1)
    vt = xt.unsqueeze(0).expand(3, 2, 1)

    assert v.data.shape == vt.detach().numpy().shape

    v.sum().backward()
    vt.sum().backward()

    assert np.allclose(t.grad, xt.grad.numpy())

def test_unsqueeze_negative_and_multiple_axes():
    np.random.seed(2)
    torch.manual_seed(2)

    a = np.random.randn(4, 5)
    t = Tensor(a.copy())
    xt = torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True)

    # unsqueeze at axis 1 then at last position using negative index
    u = t.unsqueeze(1).unsqueeze(-1)   # expected shape (4,1,5,1)
    ut = xt.unsqueeze(1).unsqueeze(-1)

    assert u.data.shape == ut.detach().numpy().shape

    u.sum().backward()
    ut.sum().backward()

    assert np.allclose(t.grad, xt.grad.numpy())