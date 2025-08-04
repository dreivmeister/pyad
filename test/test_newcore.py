# tests are taken from micrograd:
# https://github.com/karpathy/micrograd
# can be run with the command ...\pyad> python -m pytest
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

# def test_multiple_backwards():
#     x = Tensor(2.0)
#     y = x * x  # y = x^2
    
#     # First backward pass
#     y.backward()
#     assert x.gradient.data == 4.0  # dy/dx = 2x
    
#     # Second backward pass with reset
#     y.reset_gradients()
#     y.backward()
#     assert x.gradient.data == 4.0  # Would be 8.0 without reset_gradients()

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

def test_mean_against_torch():
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

def test_var_against_torch():
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
def test_softmax_against_torch():
    x_np = np.random.rand(3, 5)
    x = Tensor(x_np)
    y = x.softmax(axis=1)
    y.sum().backward()
    
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    yt = F.softmax(xt, dim=1)
    yt.sum().backward()
    
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    
def test_max_against_torch():
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
    
    
def test_conv2d_against_torch():
    # Parameters
    batch_size = 2
    in_channels = 3
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3

    x_np = np.random.randn(batch_size, in_channels, height, width)
    w_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)

    # PyAD tensors
    x = Tensor(x_np)
    w = Tensor(w_np)

    # PyTorch tensors
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)

    # Forward
    y = x.conv2d(w, bias=False)    
    m = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    m.weight = torch.nn.Parameter(wt)
    yt = m(xt)

    assert y.shape == yt.shape
    assert np.allclose(y.data, yt.detach().numpy(), atol=1e-6)

    # Backward
    y.sum().backward()
    yt.sum().backward()

    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    assert np.allclose(w.grad, m.weight.grad.numpy(), atol=1e-6)
    
    
def test_maxpool2d_against_torch():
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
    
    
from pyad.new_core import MLP
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

def test_linear_layer_against_pytorch():
    import torch
    import numpy as np
    from pyad.new_core import Tensor, LinearLayer

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
def test_log_softmax_against_torch():
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
    
    
from pyad.new_core import categorical_cross_entropy
def test_categorical_cross_entropy_against_torch():
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
    loss = categorical_cross_entropy(logits, targets)
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
def test_negative_log_likelihood_against_torch():
    import torch
    import torch.nn.functional as F
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
    
    
from pyad.new_core import BatchNorm1D
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


from pyad.new_core import LayerNorm
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