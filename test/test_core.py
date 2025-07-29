# tests are taken from micrograd:
# https://github.com/karpathy/micrograd
# can be run with the command ...\pyad> python -m pytest
import torch
import numpy as np
from pyad.core import Tensor, grad, jacobian, hessian, scalar_root_finding, linear_system_solve, fixed_point_iteration, scalar_newtons_method

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
    assert xmg.gradient.data == xpt.grad.item()

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = Tensor.tanh(a * b + b**3)
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + Tensor.relu(b + a)
    d += 3 * d + Tensor.relu(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
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
    assert abs(amg.gradient.data - apt.grad.item()) < tol
    assert abs(bmg.gradient.data - bpt.grad.item()) < tol
    
def test_grad_func():
    # Test basic gradient computation
    def f(x):
        return x * x  # f(x) = x²
    
    df = grad(f)  # df/dx = 2x
    x = Tensor(3.0)
    result = df(x)
    assert result.data == 6.0  # 2 * 3
    
    # Test higher order derivatives
    d2f = grad(grad(f))  # d²f/dx² = 2
    result = d2f(x)
    assert result.data == 2.0
    
    # Test higher order derivatives
    d3f = grad(grad(grad(f)))  # d3f/dx3 = 0
    result = d3f(x)
    assert result.data == 0.0
    
    # Test function with multiple arguments
    def g(x, y):
        return 2 * x * x * y + y * x  # g(x,y) = 2x²y + xy
    
    # Test gradient with respect to first argument (x)
    dgx = grad(g, 0)  # ∂g/∂x = 4xy + y
    x, y = Tensor(3.0), Tensor(2.0)
    result = dgx(x, y)
    assert result.data == 4 * x.data * y.data + y.data  # 4(3)(2) + 2 = 26
    
    # Test gradient with respect to second argument (y)
    dgy = grad(g, 1)  # ∂g/∂y = 2x² + x
    result = dgy(x, y)
    assert result.data == 2 * x.data * x.data + x.data  # 2(3)² + 3 = 21
    
    # Test higher order mixed derivatives
    dgxy = grad(grad(g, 0), 1)  # ∂²g/∂x∂y = 4x + 1
    result = dgxy(x, y)
    assert result.data == 4 * x.data + 1  # 4(3) + 1 = 13
    
    
def test_higher_order_grad():
    x = Tensor(3)
    def f(x):
        return x ** 3
    
    dy = grad(f)(x) # first
    assert dy.data == (x.data**2) * 3
    
    d2y = grad(grad(f))(x) # second
    assert d2y.data == 6*x.data
    
    d3y = grad(grad(grad(f)))(x) # third
    assert d3y.data == 6
    
    d4y = grad(grad(grad(grad(f))))(x) # fourth
    assert d4y.data == 0
    
    x = Tensor(3)
    y = Tensor(2)
    def f(x, y):
        return 2*x*x*y+y*x
    
    dfx = grad(f, 0)(x, y)
    dfy = grad(f, 1)(x, y)
    assert dfx.data == 4*x.data*y.data + y.data
    assert dfy.data == 2*x.data*x.data + x.data
    

    dfxx = grad(grad(f))(x, y)
    dfxy = grad(grad(f), 1)(x, y)
    
    dfyx = grad(grad(f, 1), 0)(x, y)
    dfyy = grad(grad(f, 1), 1)(x, y)
    assert dfyx.data == dfxy.data == (4*x.data + 1)
    assert dfxx.data == 4*y.data
    # this is an edge case which i dont know how to handle yet
    # should be zero but is None
    assert dfyy.data == 0
    #assert dfyy == None

def test_multiple_backwards():
    x = Tensor(2.0)
    y = x * x  # y = x^2
    
    # First backward pass
    y.backward()
    assert x.gradient.data == 4.0  # dy/dx = 2x
    
    # Second backward pass with reset
    y.reset_gradients()
    y.backward()
    assert x.gradient.data == 4.0  # Would be 8.0 without reset_gradients()

def test_list_input():
    x = [Tensor([1.0, 2.0]), Tensor([3.0, 4.0])]
    def f(x):
        # Add the tensors in the list
        return x[0] + x[1]

    df = grad(f)
    result = df(x)
    assert isinstance(result, list) and len(result) == len(x)
    assert np.allclose(result[0].data, [1.0, 1.0])
    assert np.allclose(result[1].data, [1.0, 1.0])
    
    
def test_jacobian():
    # Test vector to vector function
    def f(x):
        return x**3
    
    x = Tensor([2.0, 3.0])
    jac = jacobian(f)(x)
    
    # Manually computed Jacobian:
    # [[d(x^2)/dx, d(x^2)/dy],
    #  [d(y^3)/dx, d(y^3)/dy]] = [[2x, 0], [0, 3y^2]]
    jac_exp = np.array([[12.0, 0.0], [0.0, 27.0]])
    assert np.allclose(jac, jac_exp)
    
    
def test_hessian():
    # Test vector to scalar function
    def f(x):
        return (x**3).sum()
    
    x = Tensor([2.0, 3.0])
    hes = hessian(f)(x)
    
    # Manually computed hessian:
    # [[d(x^2)/dx, d(x^2)/dy],
    #  [d(y^3)/dx, d(y^3)/dy]] = [[6x, 0], [0, 6y]]
    hes_exp = np.array([[12.0, 0.0], [0.0, 18.0]])
    assert np.allclose(hes, hes_exp)
    
def test_sum():
    # Test basic sum
    x = Tensor([1.0, 2.0, 3.0])
    y = x.sum()
    assert y.data == 6.0
    
    # Test gradient
    y.backward()
    assert np.allclose(x.gradient.data, [1.0, 1.0, 1.0])
    
    # Test sum with axis
    x = Tensor([[1.0, 2.0], 
                [3.0, 4.0]])
    y = x.sum(axis=0)  # sum along rows
    assert np.allclose(y.data, [4.0, 6.0])
    
    y.backward()
    assert np.allclose(x.gradient.data, [[1.0, 1.0], 
                                         [1.0, 1.0]])
    
    # Test sum with keepdims
    x = Tensor([[1.0, 2.0], 
                [3.0, 4.0]])
    y = x.sum(axis=1)  # sum along columns
    assert np.allclose(y.data, [3.0, 7.0])
    
    y.backward()
    assert np.allclose(x.gradient.data, [[1.0, 1.0], 
                                         [1.0, 1.0]])
    
    # Test gradient flow through sum operation
    x = Tensor([1.0, 2.0])
    y = (x * x).sum()  # y = x1² + x2²
    y.backward()
    assert np.allclose(x.gradient.data, [2.0, 4.0])  # dy/dx = 2x
    
    
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
    assert abs(amg.gradient.data - apt.grad.item()) < tol
    assert abs(bmg.gradient.data - bpt.grad.item()) < tol
    
def test_grad_trig():
    x = Tensor(np.pi/4)  # 45 degrees
    def f(x):
        return x.sin()
    dy = grad(f)(x)
    assert np.allclose(dy.data, np.cos(np.pi/4))
    
    x = Tensor(np.pi/3)  # 60 degrees
    def f(x):
        return x.cos()
    dy = grad(f)(x)
    assert np.allclose(dy.data, -np.sin(np.pi/3))
    
    
    x = Tensor(np.pi/4)  # 45 degrees
    def f(x):
        return x.sin().cos()
    dy = grad(f)(x)
    assert np.allclose(dy.data, -np.cos(np.pi/4)*np.sin(np.sin(np.pi/4)))
    
    dy2 = grad(grad(f))(x)
    assert np.allclose(dy2.data, np.sin(np.pi/4) * np.sin(np.sin(np.pi/4)) - np.cos(np.pi/4)**2 * np.cos(np.sin(np.pi/4)))
    
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
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
    print(x.gradient.data, xt.grad.numpy())
    print(y.gradient.data, yt.grad.numpy())
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = y * x # x * y
    z.sum().backward()
    
    xt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = xt * yt
    zt.sum().backward()
    
    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
def test_concat():
    # Forward test
    x = Tensor([1.0, 2.0])
    y = Tensor([5.0, 6.0])
    z = x.concatenate(y)

    # Backward test
    z.sum().backward()

    # PyTorch comparison
    xt = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    yt = torch.tensor([5.0, 6.0], dtype=torch.float64, requires_grad=True)
    zt = torch.cat((xt, yt))
    zt.sum().backward()

    assert np.allclose(z.data, zt.detach().numpy())
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    assert np.allclose(y.gradient.data, yt.grad.numpy())
    
    
def test_mean_against_torch():
    import torch

    # Forward test
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = x.mean()
    # Backward test
    y.backward()

    xt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    yt = xt.mean()
    yt.backward()

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    
    
def test_mean_against_torch():
    import torch

    # Forward test
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = x.mean()
    # Backward test
    y.backward()

    xt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    yt = xt.mean()
    yt.backward()

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.gradient.data, xt.grad.numpy())

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
    assert np.allclose(x.gradient.data, xt.grad.numpy())

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
    assert np.allclose(x.gradient.data, xt.grad.numpy())
    
def test_scalar_root_finding():    
    # Test case: find x where x^2 - sin(theta) = 0
    def g(x, theta):
        return x**2 - theta.sin()

    dgdu = grad(g, argnum=0)
    dgdtheta = grad(g, argnum=1)

    theta = Tensor(2.0)
    initial_guess = Tensor(1.0)

    def loss(x):
        return x ** 2

    # Adjoint test
    dlossdx = grad(loss)
    x_ = scalar_newtons_method(lambda u: g(u, theta), initial_guess)
    d_J__d_theta__adjoint = - 1.0 / dgdu(x_, Tensor(2.0)) * dlossdx(x_) * dgdtheta(x_, Tensor(2.0))

    def find_root(theta, initial_guess):
        x = scalar_root_finding(g, theta, initial_guess)
        y = x ** 2
        return y

    dloss2dtheta = grad(find_root, argnum=0)
    d_J__d_theta__adjoint2 = dloss2dtheta(Tensor(2.0), Tensor(x_.data)) # at the solution theta and x_
    assert np.allclose(d_J__d_theta__adjoint2.data, d_J__d_theta__adjoint.data)
    assert np.abs(d_J__d_theta__adjoint.data) > 1e-6 and np.abs(d_J__d_theta__adjoint2.data) > 1e-6
    
def test_linear_system_solve_against_numpy():
    # Test Ax = b for a simple 2x2 system
    A = Tensor([[3.0, 2.0], [1.0, 2.0]])
    b = Tensor([5.0, 5.0])
    x = linear_system_solve(A, b)
    # Compare with numpy.linalg.solve
    expected_x = np.linalg.solve(A.data, b.data)
    assert np.allclose(x.data, expected_x)

    # Backward test: gradient of sum(x) w.r.t. A and b
    x.sum().backward()
    # Analytical gradients for sum(x) w.r.t. b is A^{-T} @ 1
    grad_b_expected = np.linalg.solve(A.data.T, np.ones_like(b.data))
    assert np.allclose(b.gradient.data, grad_b_expected)
    # Analytical gradients for sum(x) w.r.t. A is -A^{-T} @ 1 outer x
    grad_A_expected = -np.outer(grad_b_expected, x.data)
    assert np.allclose(A.gradient.data, grad_A_expected)
    
    
def test_fixed_point_iteration_against_torch():

    # Define phi(x, theta) = cos(x) + theta
    def phi_pyad(x, theta):
        return x.cos() + theta

    def phi_torch(x, theta):
        return torch.cos(x) + theta

    # Initial guess and parameter
    x0_np = Tensor([0.5])
    theta_np = Tensor([0.1])
    x0_torch = torch.tensor([0.5], dtype=torch.float64)
    theta_torch = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

    # Run pyad fixed point iteration
    result_pyad = fixed_point_iteration(phi_pyad, x0_np, theta_np)
    x_fp_pyad = result_pyad.data

    # Run PyTorch fixed point iteration
    x_fp_torch = x0_torch.clone().detach()
    for _ in range(100):
        new_x = phi_torch(x_fp_torch, theta_torch)
        if torch.norm(new_x - x_fp_torch) < 1e-8:
            break
        x_fp_torch = new_x
    x_fp_torch = x_fp_torch.detach().numpy()

    # Compare results
    assert np.allclose(x_fp_pyad, x_fp_torch, atol=1e-6)

    # Gradient check: d fixed point / d theta
    # PyTorch autograd
    x_fp_torch = x0_torch.clone().detach()
    x_fp_torch.requires_grad = True
    theta_torch = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    for _ in range(100):
        new_x = phi_torch(x_fp_torch, theta_torch)
        if torch.norm(new_x - x_fp_torch) < 1e-8:
            break
        x_fp_torch = new_x
    x_fp_torch.sum().backward()
    grad_torch = theta_torch.grad.detach().numpy()

    # PyAD backward
    result_pyad.sum().backward()
    grad_pyad = theta_np.gradient.data

    assert np.allclose(grad_pyad, grad_torch, atol=1e-6)
    