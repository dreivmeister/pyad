from core import grad, Tensor, scalar_root_finding, scalar_newtons_method

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
assert np.abs(d_J__d_theta__adjoint.data) > 1e-6 and np.abs(theta.gradient.data)