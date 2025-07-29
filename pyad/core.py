# contains the Tensor class which wraps a np array and provides gradient computation
import numpy as np
from scipy.linalg import lu, solve_triangular
from scipy.signal import convolve2d, correlate2d
from math import prod

def promote_array_to_tensor(array):
    return array if isinstance(array, Tensor) else Tensor(array)

def shape_to_axis(old_shape, new_shape):
    return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

def argsort(x): 
    return type(x)(sorted(range(len(x)), key=x.__getitem__)) 

class Tensor:
    def __init__(self, data, prev_tensors=(), dtype=np.float64):
        # Unwrap if data is already a Tensor
        while isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=dtype) # actual value
        self.prev_tensors = prev_tensors # tensors that were used to compute this tensor
        self.gradient = None # gradient of this tensor
        self.backw_op = lambda x: None # function to compute the gradient of this tensor
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.gradient})"
    
    def reset_gradients(self):
        """Resets the gradient of this tensor and all tensors in the computation graph."""
        visited_set = set()
        def reset(tensor):
            if tensor not in visited_set and tensor is not None:
                visited_set.add(tensor)
                if isinstance(tensor, Tensor):
                    if tensor.gradient is not None:
                        reset(tensor.gradient)  # Reset gradients of gradients
                    tensor.gradient = None
                    
                    for child in tensor.prev_tensors:
                        reset(child)
        reset(self)
    
    def backward(self, gradient=None):
        if gradient is None:
            gradient = Tensor(np.ones_like(self.data, dtype=np.float64))
        elif not isinstance(gradient, Tensor):
            gradient = promote_array_to_tensor(gradient)
        self.gradient = gradient

        # Calculate topological order and initialize gradients
        topological_order = []
        visited_set = set()

        def toposort(tensor):
            if tensor not in visited_set:
                visited_set.add(tensor)
                if tensor != self:
                    tensor.gradient = Tensor(np.zeros_like(tensor.data, dtype=np.float64))
                for child in tensor.prev_tensors:
                    toposort(child)
                topological_order.append(tensor)
        toposort(self)

        # Backward pass
        for tensor in reversed(topological_order):
            tensor.backw_op(tensor.gradient)
    
    def __mul__(self, other):
        other = promote_array_to_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def backw_op(gradient):
            if self.data.ndim == 0: # scalar * vector
                self.gradient += gradient.dot(other)
            else:
                self.gradient += gradient * other
            
            if other.data.ndim == 0: # vector * scalar
                other.gradient += gradient.dot(self)
            else:
                other.gradient += gradient * self
        out.backw_op = backw_op
        return out

    def __add__(self, other):
        other = promote_array_to_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def backw_op(gradient):
            self.gradient += gradient
            other.gradient += gradient
        out.backw_op = backw_op
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,))

        def backw_op(gradient):
            self.gradient += (other * self**(other-1)) * gradient
        out.backw_op = backw_op
        return out
    
    def transpose(self, order=None):
        if order is None:
            order = tuple(range(self.data.ndim))[::-1]
        input_order = order
        out = Tensor(np.transpose(self.data, order), (self,))
        
        def backw_op(gradient):
            self.gradient += gradient.transpose(argsort(input_order))
            # or: self.grad += np.transpose(gradient, np.argsort(input_order))
        out.backw_op = backw_op

        return out

    def reshape(self, new_shape):
        out = Tensor(self.data.reshape(new_shape), (self,))

        def backw_op(gradient):
            self.gradient += gradient.reshape(self.data.shape)
        out.backw_op = backw_op
        
        return out
    
    def outer(self, other):
        other = promote_array_to_tensor(other)
        out = Tensor(np.outer(self.data, other.data), (self, other))

        def backw_op(gradient):
            # gradient is shape (len(self), len(other))
            self.gradient += gradient.dot(other)
            other.gradient += self.dot(gradient)
        out.backw_op = backw_op

        return out
    
    def dot(self, other):
        # this allows matrix multiply, matrix vector multiply and dot product
        other = promote_array_to_tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def backw_op(gradient):
            # Gradient w.r.t. self
            if other.data.ndim == 2:
                grad_self = gradient.dot(other.transpose())
            elif other.data.ndim == 1 and gradient.data.ndim == 1:
                grad_self = gradient.outer(other)
            else:
                grad_self = gradient * other
            self.gradient += grad_self.reshape(self.data.shape)

            # Gradient w.r.t. other
            if self.data.ndim == 2:
                grad_other = self.transpose().dot(gradient)
            elif self.data.ndim == 1 and gradient.data.ndim == 1:
                grad_other = self.outer(gradient)
            else:
                grad_other = self * gradient
            other.gradient += grad_other.reshape(other.data.shape)
        out.backw_op = backw_op
        
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), (self,))
        
        def backw_op(gradient):
            self.gradient += self.cos() * gradient
        out.backw_op = backw_op
        
        return out
    
    def cos(self):
        out = Tensor(np.cos(self.data), (self,))
        
        def backw_op(gradient):
            self.gradient += (-self.sin()) * gradient
        out.backw_op = backw_op
        
        return out

    @staticmethod
    def relu(self):
        # only first derivative
        out = Tensor(self.data * (self.data > 0), (self,))

        def backw_op(gradient):
            self.gradient += (self.data > 0) * gradient.data
        out.backw_op = backw_op
        
        return out

    @staticmethod
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,))

        def backw_op(gradient):
            self.gradient += (1 - out**2) * gradient
        out.backw_op = backw_op
        
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def backw_op(gradient):
            self.gradient += out * gradient
        out.backw_op = backw_op
        
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def backw_op(gradient):
            self.gradient += (1.0/self) * gradient
        out.backw_op = backw_op
        
        return out
    
    def sqrt(self):
        return self ** (1/2)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def broadcast_to(self, shape):
        out = Tensor(np.broadcast_to(self.data, shape), (self,))

        def backw_op(gradient):
            self.gradient += gradient.sum(shape_to_axis(gradient.data.shape,self.data.shape))
        out.backw_op = backw_op
        
        return out
    
    def sum(self, axis=None):
        out = Tensor(np.sum(self.data, axis=axis), (self,))

        def backw_op(gradient):
            self.gradient += gradient.broadcast_to(self.data.shape)
        out.backw_op = backw_op
        
        return out
    
    # only first derivative
    def mean(self, axis=None):
        out = Tensor(np.mean(self.data, axis=axis), (self,))
        def backw_op(gradient):
            # Distribute gradient equally to all elements
            grad = gradient.data
            if axis is not None:
                # Broadcast grad to input shape
                grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.data.shape)
                count = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])
            else:
                count = self.data.size
                grad = np.broadcast_to(grad, self.data.shape)
            self.gradient += grad / count
        out.backw_op = backw_op
        return out
    
    # only first derivative
    def var(self, axis=None, unbiased=False):
        m = self.data.mean(axis=axis)
        out = Tensor(np.mean((self.data - m) ** 2, axis=axis), (self,))
        def backw_op(gradient):
            grad = gradient.data
            if axis is not None:
                grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.data.shape)
                count = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])
            else:
                count = self.data.size
                grad = np.broadcast_to(grad, self.data.shape)
            # d/dx mean((x-mean(x))^2) = 2*(x-mean(x))/N
            self.gradient += 2 * (self.data - m.data) * grad / count
        out.backw_op = backw_op
        return out
    
    def concatenate(self, other):
        # only gradient not higher order
        other = promote_array_to_tensor(other)
        if self.data.ndim > 1 or other.data.ndim > 1:
            raise ValueError("concat only supports 1D tensors")
        out = Tensor(np.concatenate((self.data, other.data)), (self, other))

        def backw_op(gradient):
            self.gradient += gradient.data[:self.data.shape[0]].reshape(self.data.shape)
            other.gradient += gradient.data[self.data.shape[0]:].reshape(other.data.shape)
        out.backw_op = backw_op
        
        return out
    
    def pad(self, pad_width):
        if self.data.ndim > 1:
            raise ValueError("pad only supports 1D tensors")
        if len(pad_width) != 2:
            raise ValueError("pad_width must be a tuple of length 2")
        # only gradient not higher order
        out = Tensor(np.pad(self.data, pad_width), (self,))

        def backw_op(gradient):
            self.gradient += gradient.data[pad_width[0]:-pad_width[1]].reshape(self.data.shape)
        out.backw_op = backw_op
        
        return out
    
    def flatten(self):
        out = Tensor(self.data.flatten(), (self,))

        def backw_op(gradient):
            self.gradient += gradient.reshape(self.data.shape)
        out.backw_op = backw_op
        
        return out
    
    @staticmethod
    def conv2d(x, kernels, output_shape):
        # only stride=1 and valid padding
        #https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py
        # x is the input image
        
        batch_size, out_channels, out_height, out_width = output_shape
        out = np.random.randn(*output_shape) # bias
        
        in_channels = x.shape[1]
        for k in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    out[k,i] += correlate2d(x.data[k,j], kernels.data[i,j], "valid")
        out = Tensor(out, (x, kernels))
        
        def backw_op(gradient):
            x.gradient = np.zeros_like(x.data)
            kernels.gradient = np.zeros_like(kernels.data)

            for k in range(batch_size):
                for i in range(out_channels):
                    for j in range(in_channels):
                        kernels.gradient[i,j] += correlate2d(x.data[k,j], gradient[k,i], "valid")
                        x.gradient[k,j] += convolve2d(gradient[k,i], kernels.data[i,j], "full")
        out.backw_op = backw_op
        
        return out
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
    
# some implicit autodiff rules taken from:
# https://fkoehler.site/implicit-autodiff-table/
def scalar_newtons_method(f, curr_val, max_iter=100, tol=1e-8):
    # x_k+1 = x_k - (f(x_k)/f'(x_k))    
    # scalar version
    f_prime = grad(f)
    
    for _ in range(max_iter):
        new_val = curr_val - (f(curr_val) / (f_prime(curr_val) + 1e-10))
        if abs(new_val.data - curr_val.data) < tol:
            break
        curr_val = new_val
    return curr_val
    
def scalar_root_finding(g, theta, initial_guess):
    # forward pass to find the root
    x = scalar_newtons_method(lambda u: g(u, theta), initial_guess)
    dg_dx = grad(g, argnum=0)
    dg_dtheta = grad(g, argnum=1)
    
    out = Tensor(x.data, (initial_guess, theta))
    
    def backw_op(gradient):
        theta_ = Tensor(np.copy(theta.data))
        theta.gradient -= 1.0 / dg_dx(x, theta_) * dg_dtheta(x, theta_) * gradient
    out.backw_op = backw_op
    
    return out

def fixed_point_iteration_helper(phi, curr_val, theta, max_iter=100, tol=1e-8):
    # x_k+1 = phi(x_k)
    # vector version
    for _ in range(max_iter):
        new_val = phi(curr_val, theta)
        if np.linalg.norm(new_val.data - curr_val.data) < tol:
            break
        curr_val = new_val
    return curr_val

def fixed_point_iteration(phi, initial_guess, theta):
    # forward pass to find the fixed point
    # x0 = initial_guess
    x = fixed_point_iteration_helper(phi, initial_guess, theta)
    dphi_dx = jacobian(lambda u: phi(u, theta)) # theta is constant
    dphi_dtheta = jacobian(lambda u: phi(x, u)) # x is constant
    
    out = Tensor(x.data, (initial_guess, theta))
    
    def backw_op(gradient):
        theta_ = Tensor(np.copy(theta.data))
        x_ = Tensor(np.copy(x.data))
        lambda_ = linear_system_solve_helper(((np.eye(x_.data.shape[0]) - dphi_dx(x_))).transpose().data, gradient.data)
        theta.gradient -= dphi_dtheta(theta_).transpose().data @ lambda_
    out.backw_op = backw_op
    
    return out

def linear_system_solve_helper(A, b):
    P, L, U = lu(A)
    P_Tb = P.T @ b
    y = solve_triangular(L, P_Tb, lower=True, unit_diagonal=True)
    return solve_triangular(U, y, lower=False)
    

def linear_system_solve(A, b):
    # A is a square matrix
    if A.data.ndim != 2 or A.data.shape[0] != A.data.shape[1]:
        raise ValueError("A must be a square matrix")
    if b.data.ndim != 1 or b.data.shape[0] != A.data.shape[0]:
        raise ValueError("b must be a vector of the same size as A")
    
    out = Tensor(linear_system_solve_helper(A.data, b.data), (A, b))
    
    def backw_op(gradient):
        # gradient is shape (len(b), len(A))
        theta = linear_system_solve(A.transpose(), gradient)
        A.gradient += -theta.outer(out)
        b.gradient += theta
    out.backw_op = backw_op
    
    return out

"""
def vector_newtons_method(f, curr_val, max_iter=100, tol=1e-8):
    # x_k+1 = x_k - (f(x_k)/f'(x_k))    
    # vector version
    f_prime = jacobian(f)
    
    for _ in range(max_iter):
        correction_term = linear_system_solve_helper(f_prime(curr_val).data, (-f(curr_val)).data)
        new_val = correction_term + curr_val
        if np.linalg.norm(new_val.data - curr_val.data) < tol:
            break
        curr_val = new_val
    return curr_val

def nonlinear_system_solving(g, theta, initial_guess):
    # forward pass to find the root
    x = vector_newtons_method(lambda u: g(u, theta), initial_guess)
    dg_dx = jacobian(lambda u: g(u, theta)) # theta is constant
    dg_dtheta = jacobian(lambda u: g(x, u)) # x is constant
    
    out = Tensor(x.data, (x, theta))
    
    def backw_op(gradient):
        theta_ = linear_system_solve_helper(dg_dx(x).transpose().data, gradient.data)
        theta.gradient -= np.dot(dg_dtheta(theta).transpose().data, theta_)
    out.backw_op = backw_op
    
    return out
"""

"""
from scipy.integrate import solve_ivp

def neural_ode(f, u0, T, theta):
    sol = solve_ivp(f, [0, T], u0.data, args=(theta,), t_eval=[T]) # only care about the final value
    if sol.success:
        out = Tensor(sol.y[:, -1], (u0, theta))
    else:
        raise RuntimeError("ODE solve failed")
    
    df_du = jacobian(lambda u: f(T, u, theta))  # Jacobian of f at time T
    df_dtheta = jacobian(lambda u: f(T, u0, u))  # Jacobian of f w.r.t. theta at time T
    
    def backw_op(gradient):
        def lambda_(t, u, theta):
            # lambda is the adjoint variable
            return -df_du(t, u, theta) * u
        sol_lambda = solve_ivp(lambda_, [T, 0], sol.y[:, -1], args=(theta,))
        adjoint = -gradient.data * sol.y[:, -1]
        # we need to compute the gradient of the integral
        def adjoint_f(t, u):
            return -f(u, theta).data * adjoint
        # we can use the same solver to compute the adjoint
        adjoint_sol = solve_ivp(adjoint_f, [T, 0], sol.y[:, -1], args=(theta,), t_eval=[0])
        if adjoint_sol.success:
            theta_ = Tensor(adjoint_sol.y[:, -1], (theta,))
            theta.gradient += theta_.data
        else:
            raise RuntimeError("Adjoint sensitivity computation failed")
    out.backw_op = backw_op
    
    return out
""" 
    
    

# the subval, vjp and grad functions are taken from autodidact and adapted handle lists of tensors:
# https://github.com/mattjj/autodidact
def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def make_vjp(fun, x):
    end_valnode = fun(x)
    
    def vjp(g):
        if end_valnode is None:
            # Handle list input
            if isinstance(x, list):
                return [Tensor(np.zeros_like(xi.data)) for xi in x]
            return Tensor(np.zeros_like(x.data))
            
        end_valnode.reset_gradients()
        end_valnode.backward(g)
        
        # Handle list input by returning list of gradients
        if isinstance(x, list):
            return [xi.gradient or Tensor(np.zeros_like(xi.data)) for xi in x]
        return x.gradient or Tensor(np.zeros_like(x.data))
    
    return vjp, end_valnode


def grad(fun, argnum=0):
    def gradfun(*args):
        unary_fun = lambda x: fun(*subval(args, argnum, x))
        vjp, ans = make_vjp(unary_fun, args[argnum])
        
        if ans is None:
            # Handle list input
            if isinstance(args[argnum], list):
                return [Tensor(np.zeros_like(xi.data)) for xi in args[argnum]]
            return Tensor(np.zeros_like(args[argnum].data))
        
        # # check if ans is a scalar
        # if ans.data.ndim != 0:
        #     raise ValueError("Gradient only defined for scalar outputs. You can use jacobian instead.")
            
        result = vjp(np.ones_like(ans.data, dtype=np.float64))
        return result
    
    return gradfun


def jacobian(fun):
    def jacfun(*args):
        # Get output shape by running function once
        #unary_fun = lambda x: fun(*subval(args, argnum, x))
        vjp, ans = make_vjp(fun, args[0])
        
        if ans is None:
            return None
        
        if not isinstance(ans, Tensor) or not isinstance(args[0], Tensor):
            raise ValueError("Function must take and return a Tensor")
            
        output_shape = ans.data.shape
        input_shape = args[0].data.shape
        
        # Initialize Jacobian matrix
        jac = np.zeros((output_shape[0], input_shape[0]))        
        # Compute each partial derivative
        for i in np.ndindex(output_shape):                
            v = np.zeros(output_shape, dtype=np.float64)
            v[i] = 1.0
            jac[i,:] = vjp(v).data
            
        return jac
    return jacfun


def hessian(fun):
    def hessfun(*args):
        # First check if function returns a scalar
        test_out = fun(*args)
        if test_out is None:
            return None
            
        if not isinstance(test_out, Tensor) or not isinstance(args[0], Tensor):
            raise ValueError("Function must take and return a Tensor")
            
        if test_out.data.size != 1:
            raise ValueError("Hessian only defined for scalar-valued functions")
            
        # Compute gradient function
        def grad_fun(x):
            # Returns vector of partial derivatives
            vjp, ans = make_vjp(fun, x)
            if ans is None:
                return None
            return vjp(np.ones_like(ans.data, dtype=np.float64))
            
        # Compute Jacobian of gradient (which gives us the Hessian)
        return jacobian(grad_fun)(*args)
        
    return hessfun