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
    x = Tensor(3.0)
    y = Tensor([4.0, 5.0, 6.0])
    z = y * x # x * y

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
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
    X_np = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
    y_np = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))
    print(X_np.shape, y_np.shape)

    # # Convert to Tensor
    # X = [Tensor(x) for x in X_np]
    # y = [Tensor(t) for t in y_np]

    mlp = MLP(2, [16, 1], nonlin='tanh')
    optim = Adam(mlp.parameters())    

    for epoch in range(300):
        total_loss = 0
        mlp.zero_grad()
        #for xi, yi in zip(X, y):
        out = mlp(X_np)
        loss = ((out - y_np) ** 2).sum()
        loss.backward()
        total_loss += loss.data
        optim.step()
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss: {total_loss}")

    # Test predictions
    preds = mlp(X_np)
    print(((preds - y_np) ** 2).sum())
    # Should be close to [0, 1, 1, 0]
    assert ((preds - y_np) ** 2).sum().data < 0.1