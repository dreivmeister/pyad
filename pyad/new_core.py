import numpy as np
from math import prod
from scipy import signal
#from graphviz import Digraph

# helpers
def shape_to_axis(old_shape, new_shape):
    assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
    return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

def unpad(x, arg):
    return x[tuple(slice(p[0], p[1], None) for p in arg)]

def stride(x, arg):
    return x[tuple(slice(None, None, i) for i in arg)]

def argfix(*x): 
    return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], (tuple, list)) else tuple(x)

def promote_array_to_tensor(array):
    return array if isinstance(array, Tensor) else Tensor(array)

# https://stackoverflow.com/questions/33823 52/equivalent-of-numpy-argsort-in-basic-python
def argsort(x): 
    return type(x)(sorted(range(len(x)), key=x.__getitem__)) 

def transpose_last_two(shape):
    return list(range(len(shape)-2))+[-1, -2]


class Tensor:    
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs):
        self.data = np.asarray(data, dtype=np.float64)
        self.prev = prev
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.op = op
        self.grad_fn = lambda x: None

    def __repr__(self):
        if self.data.ndim < 2:
            return f'Tensor(data={self.data}, grad={self.grad})'    
        return f'Tensor\ndata=\n{self.data},\ngrad=\n{self.grad})'
    
    # plot utility
    def generate_graph(self):
        dot = Digraph(comment='DAG')
        visited = set()
        def build_graph(t):
            if t not in visited:
                visited.add(t)
                if t.name:
                    nm = t.name
                    shape = "box"
                    color = ""
                else:
                    nm = t.op.__name__
                    shape = ""
                    color = "lightblue2"
                    if nm == "<lambda>":
                        nm = str(t.data)
                        
                dot.node(str(hash(t)), nm, shape=shape, color=color, style='filled')
                for p in t.prev:
                    dot.edge(str(hash(p)), str(hash(t)))
                    build_graph(p)
        build_graph(self)
        return dot

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float64)
        self.grad = gradient

        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t.prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t.grad_fn(t.grad)
    
    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def zeros_like(tensor):
        return Tensor(np.zeros_like(tensor))
    
    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))
    
    @staticmethod
    def ones_like(tensor):
        return Tensor(np.ones_like(tensor))
    
    @staticmethod
    def eye(dim):
        return Tensor(np.eye(dim))
    
    @staticmethod
    def rand(shape):
        return Tensor(np.random.rand(*shape))
    
    @staticmethod
    def randn(shape):
        return Tensor(np.random.randn(*shape))
    
    @staticmethod
    def uniform(low, high, shape):
        return Tensor(np.random.uniform(low,high,shape))
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype
    
    def xavier_uniform(self):
        fan_in = self.shape[0]
        fan_out = self.shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.data = np.random.uniform(-limit, limit, self.shape)

    def copy(self):
        new_tensor = Tensor(self.data,self.prev,self.op)
        new_tensor.grad = self.grad
        new_tensor.grad_fn = self.grad_fn
        return new_tensor
    
    def __getitem__(self, idx):
        # Normalize index to tuple and convert Tensor/list indices to numpy arrays
        if not isinstance(idx, tuple):
            idx = (idx,)
        norm_idx = []
        advanced = False
        for i in idx:
            ii = i
            if isinstance(ii, Tensor):
                ii = ii.data
            if isinstance(ii, list):
                ii = np.array(ii)
            # detect advanced indexing (integer/boolean array with ndim>0)
            if isinstance(ii, np.ndarray) and ii.ndim > 0 and (np.issubdtype(ii.dtype, np.integer) or ii.dtype == bool):
                advanced = True
            norm_idx.append(ii)

        y_data = self.data[tuple(norm_idx)]
        out = Tensor(y_data, (self,), op=Tensor.__getitem__)

        def grad_fn(gradient):
            # scatter-add gradient back to self.grad at the selected indices
            if advanced:
                # use unbuffered accumulation for advanced indexing (handles repeated indices correctly)
                np.add.at(self.grad, tuple(norm_idx), gradient)
            else:
                # regular slicing / ints / ellipsis / None
                self.grad[tuple(norm_idx)] += gradient
        out.grad_fn = grad_fn
        return out
    
    def __setitem__(self, idx, value):
        # Ensure value is a Tensor
        value = value if isinstance(value, Tensor) else Tensor(value)

        # Normalize index to tuple and detect advanced indexing
        if not isinstance(idx, tuple):
            idx = (idx,)
        norm_idx = []
        advanced = False
        for i in idx:
            ii = i
            if isinstance(ii, Tensor):
                ii = ii.data
            if isinstance(ii, list):
                ii = np.array(ii)
            if isinstance(ii, np.ndarray) and ii.ndim > 0 and (np.issubdtype(ii.dtype, np.integer) or ii.dtype == bool):
                advanced = True
            norm_idx.append(ii)
        norm_idx = tuple(norm_idx)

        # Perform the in-place data assignment (NumPy handles broadcasting)
        self.data[norm_idx] = value.data

        # Chain autograd: keep previous parents and add 'value' as a new parent
        old_prev = self.prev
        old_grad_fn = self.grad_fn
        self.prev = tuple(old_prev) + (value,)
        self.op = Tensor.__setitem__

        def grad_fn(gradient):
            # Split the upstream gradient:
            # - Part for self: zero out the region that was overwritten
            # - Part for value: the gradient of the assigned slice (with broadcasting reduction)
            grad_to_self = np.array(gradient, copy=True)
            # Zero out the assigned slice for self's chain
            if advanced:
                # For advanced indexing, set the selected region to zero
                grad_to_self[norm_idx] = 0
            else:
                grad_to_self[norm_idx] = 0

            # Propagate to self's original parents
            if old_grad_fn is not None:
                old_grad_fn(grad_to_self)

            # Propagate to value
            grad_to_value = gradient[norm_idx]
            # Reduce over broadcasted dims so it matches value.data shape
            grad_to_value = Tensor.checkbroadcast(value, grad_to_value)
            value.grad += grad_to_value

        self.grad_fn = grad_fn
    
    @staticmethod
    def checkbroadcast(a, grad_a):
        if grad_a.shape != a.data.shape:
            s_shape = a.data.shape
            g_shape = grad_a.shape
            if len(s_shape) < len(g_shape):
                s_shape = (1,) * (len(g_shape) - len(s_shape)) + s_shape
            axes = tuple(i for i, (s, g) in enumerate(zip(s_shape, g_shape)) if s == 1 and g > 1)
            if axes:
                grad_a = grad_a.sum(axis=axes, keepdims=True)
            grad_a = grad_a.reshape(a.data.shape)
        return grad_a

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), op=self.__add__)
        def grad_fn(gradient):
            # Gradient w.r.t. self
            grad_self = gradient
            # if shape mismatch, then self was broadcasted during the operation
            # adjust grad_self accordingly
            grad_self = Tensor.checkbroadcast(self, grad_self)
            self.grad += grad_self

            # Gradient w.r.t. other
            grad_other = gradient
            grad_other = Tensor.checkbroadcast(other, grad_other)
            other.grad += grad_other
        out.grad_fn = grad_fn
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), op=self.__mul__)

        def grad_fn(gradient):
            # Gradient w.r.t. self
            grad_self = gradient * other.data
            grad_self = Tensor.checkbroadcast(self, grad_self)
            self.grad += grad_self

            # Gradient w.r.t. other
            grad_other = gradient * self.data
            grad_other = Tensor.checkbroadcast(other, grad_other)
            other.grad += grad_other

        out.grad_fn = grad_fn
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), op=self.__pow__)
        
        def grad_fn(gradient):
            self.grad += gradient * (other * (self.data ** (other-1)))
        out.grad_fn = grad_fn
        
        return out
    
    def sqrt(self):
        return self ** (1/2)
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __neg__(self):
        return self * -1
    
    def sum(self, axis=None, keepdims=False):
        input_shape = self.data.shape
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), op=self.sum)
        def grad_fn(gradient):
            # Always broadcast gradient to input shape
            self.grad += np.broadcast_to(np.expand_dims(gradient, axis) if axis is not None and not keepdims else gradient, input_shape)
        out.grad_fn = grad_fn
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), op=self.mean)
        # Compute n: number of elements reduced per output element
        if axis is None:
            n = self.data.size
            axes = tuple(range(self.data.ndim))
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = np.prod([self.data.shape[ax] for ax in axes])

        def grad_fn(gradient):
            # Broadcast gradient to input shape
            grad = gradient
            if not keepdims and axis is not None:
                # Insert singleton dims for reduced axes
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            grad = np.ones_like(self.data) * grad / n
            self.grad += grad
        out.grad_fn = grad_fn
        return out
    
    def var(self, axis=None, keepdims=False):
        out = Tensor(np.var(self.data, axis=axis, keepdims=keepdims), (self,), op=self.var)
        m = np.mean(self.data, axis=axis, keepdims=True)
        if axis is None:
            n = self.data.size
            axes = tuple(range(self.data.ndim))
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = np.prod([self.data.shape[ax] for ax in axes])

        def grad_fn(gradient):
            grad = gradient
            if not keepdims and axis is not None:
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            grad = grad * (2.0 / n) * (self.data - m)
            self.grad += grad
        out.grad_fn = grad_fn
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), (self,), op=self.sin)
        
        def grad_fn(gradient):
            self.grad += gradient * np.cos(self.data)
        out.grad_fn = grad_fn
        
        return out
    
    def cos(self):
        out = Tensor(np.cos(self.data), (self,), op=self.cos)
        
        def grad_fn(gradient):
            self.grad += gradient * -np.sin(self.data)
        out.grad_fn = grad_fn
        
        return out
                
    def relu(self):
        out = Tensor(self.data*(self.data>0), (self,), op=self.relu)
        
        def grad_fn(gradient):
            self.grad += gradient * (self.data > 0)
        out.grad_fn = grad_fn
        
        return out
    
    def sigmoid(self):
        out = Tensor(1.0 / (1 + np.exp(-self.data)), (self,), op=self.sigmoid)
        
        def grad_fn(gradient):
            self.grad += gradient * (out.data * (1 - out.data))
        out.grad_fn = grad_fn
        
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), op=self.tanh)
        def grad_fn(gradient):
            self.grad += gradient * (1 - t**2)
        out.grad_fn = grad_fn
        
        return out
    
    def gelu(self):
        # Approximate GELU (Hendrycks & Gimpel) using tanh
        # gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715 x^3) ))
        x = self.data
        c0 = 0.044715
        c1 = 0.7978845608028654  # sqrt(2/pi)
        inner = c1 * (x + c0 * x**3)
        tanh_inner = np.tanh(inner)
        gelu_data = 0.5 * x * (1 + tanh_inner)
        out = Tensor(gelu_data, (self,), op=self.gelu)
        def grad_fn(gradient):
            # d/dx gelu(x) ≈ 0.5*(1 + tanh_inner) + 0.5*x*(1 - tanh_inner^2)*c1*(1 + 3*c0*x^2)
            deriv = 0.5*(1 + tanh_inner) + 0.5 * x * (1 - tanh_inner**2) * c1 * (1 + 3*c0*x**2)
            self.grad += gradient * deriv
        out.grad_fn = grad_fn
        
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), op=self.exp)
        
        def grad_fn(gradient):
            self.grad += gradient * out.data
        out.grad_fn = grad_fn
        
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,), op=self.log)
        
        def grad_fn(gradient):
            self.grad += gradient * (1. / self.data)
        out.grad_fn = grad_fn
        
        return out

    def max(self, axis=None, keepdims=True):
        m = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(m, (self,), op=self.max)
        
        def grad_fn(gradient):
            self.grad += gradient * np.equal(self.data, m).astype(np.float32)
        out.grad_fn = grad_fn
        
        return out
    
    def reshape(self, new_shape):
        out = Tensor(np.reshape(self.data, new_shape), (self,), op=self.reshape)
        
        def grad_fn(gradient):
            self.grad += np.reshape(gradient, self.data.shape)
        out.grad_fn = grad_fn
        
        return out
    
    def transpose(self, order):
        out = Tensor(np.transpose(self.data, order), (self,), op=self.transpose)
        
        def grad_fn(gradient):
            self.grad += np.transpose(gradient, argsort(order))
        out.grad_fn = grad_fn

        return out
    
    @staticmethod
    def concatenate(seq, axis = 0):
        # seq is a tuple of Tensors which should be concated
        # along axis (axis=0 - along rows), (axis=1 - along cols)
        # assume seq only contains 2d arrays
        n = len(seq)
        out = Tensor(np.concatenate([a.data for a in seq], axis=axis), (*seq,), op=Tensor.concatenate)
        
        def grad_fn(gradient):
            # gradient is of shape out
            # Build cumulative split points along axis
            sizes = [a.shape[axis] for a in seq]
            split_points = np.cumsum(sizes)[:-1]  # exclude last
            gradient_split = np.split(gradient, split_points, axis=axis)
            for i in range(n):
                seq[i].grad += gradient_split[i]
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def stack(seq, axis=0):
        # seq is a tuple/list of Tensors which should be stacked along a new axis
        out_data = np.stack([a.data for a in seq], axis=axis)
        out = Tensor(out_data, (*seq,), op=Tensor.stack)

        # normalize axis to positive index w.r.t. out_data
        axis_norm = axis if axis >= 0 else out_data.ndim + axis

        def grad_fn(gradient):
            # gradient has shape equal to out_data
            # for each i, take the slice along the new axis and add to the corresponding input grad
            for i, a in enumerate(seq):
                slc = [slice(None)] * gradient.ndim
                slc[axis_norm] = i
                a.grad += gradient[tuple(slc)]
        out.grad_fn = grad_fn

        return out
    
    
    def expand(self, *sizes):
        # torch-like expand: -1 keeps size, can add trailing dims, only expand dims of size 1
        orig = self.data.shape
        if len(sizes) < len(orig):
            raise AssertionError("expand: target dims must be >= input dims")
        # align original dimensions to the LEFT (torch semantics when new dims are appended)
        padded = tuple(orig) + (1,) * (len(sizes) - len(orig))

        target = []
        for s_in, s_out in zip(padded, sizes):
            if s_out == -1:
                target.append(s_in)
            else:
                assert s_out == s_in or s_in == 1, f"expand: cannot expand {s_in} -> {s_out}"
                target.append(int(s_out))

        out_data = np.broadcast_to(self.data.reshape(padded), tuple(target))
        out = Tensor(out_data, (self,), op=Tensor.expand)

        # axes where broadcasting happened (s_in==1 and expanded >1), including added trailing dims
        reduce_axes = tuple(i for i, (s_in, s_out) in enumerate(zip(padded, target)) if s_in == 1 and s_out > 1)

        def grad_fn(grad):
            g = grad
            if reduce_axes:
                g = g.sum(axis=reduce_axes, keepdims=True)
            # reshape back to padded then to original
            g = g.reshape(padded).reshape(orig)
            self.grad += g

        out.grad_fn = grad_fn
        return out
    
    
    def unsqueeze(self, axis):
        """
        Insert dimension(s) of size 1 at the specified axis (int) or sequence of axes (list/tuple).
        Supports negative indices and multiple axes applied in order (like torch.unsqueeze).
        """
        # normalize axes to list
        if isinstance(axis, (list, tuple)):
            axes_in = [int(a) for a in axis]
        else:
            axes_in = [int(axis)]

        out_data = self.data
        applied_axes = []
        curr_ndim = out_data.ndim
        for a in axes_in:
            # convert negative axis relative to the current ndim (insertion can be at end)
            if a < 0:
                a = a + curr_ndim + 1
            assert 0 <= a <= curr_ndim, f"unsqueeze: axis {a} out of range for ndim {curr_ndim}"
            out_data = np.expand_dims(out_data, axis=a)
            applied_axes.append(a)
            curr_ndim += 1

        out = Tensor(out_data, (self,), op=Tensor.unsqueeze)

        def grad_fn(gradient):
            g = gradient
            # collapse inserted axes in reverse order; if axis was broadcasted later,
            # sum over that axis first to reduce to size 1, then squeeze.
            for a in reversed(applied_axes):
                if g.shape[a] != 1:
                    g = g.sum(axis=a, keepdims=True)
                g = np.squeeze(g, axis=a)
            self.grad += g

        out.grad_fn = grad_fn
        return out
    

    def dot(self, other):
        # this allows matrix multiply, matrix vector multiply and dot product
        other = promote_array_to_tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def grad_fn(gradient):
            # Gradient w.r.t. self
            if other.data.ndim == 2:
                grad_self = np.dot(gradient, other.data.T)
            elif other.data.ndim == 1 and gradient.ndim == 1:
                grad_self = np.outer(gradient, other.data)
            else:
                grad_self = gradient * other.data
            self.grad += grad_self.reshape(self.data.shape)

            # Gradient w.r.t. other
            if self.data.ndim == 2:
                grad_other = np.dot(self.data.T, gradient)
            elif self.data.ndim == 1 and gradient.ndim == 1:
                grad_other = np.outer(self.data, gradient)
            else:
                grad_other = self.data * gradient
            other.grad += grad_other.reshape(other.data.shape)
        out.grad_fn = grad_fn
        
        return out
    
    def bmm(self, other):
        # Batch matrix multiplication for 3D or 4D tensors
        other = promote_array_to_tensor(other)
        assert self.data.ndim in (3, 4) and other.data.ndim == self.data.ndim, "Both tensors must be 3D or 4D for bmm"
        # (B, N, M) @ (B, M, K) -> (B, N, K)
        # (A, B, N, M) @ (A, B, M, K) -> (A, B, N, K)
        out = Tensor(np.matmul(self.data, other.data), (self, other), op=Tensor.bmm)

        def grad_fn(gradient):
            # (B, N, K) grad
            # (A, B, N, K) grad
            self.grad += np.matmul(gradient, np.swapaxes(other.data, -2, -1))
            other.grad += np.matmul(np.swapaxes(self.data, -2, -1), gradient)
        out.grad_fn = grad_fn

        return out
    
    def outer(self, other):
        other = promote_array_to_tensor(other)
        out = Tensor(np.outer(self.data, other.data), (self, other))

        def grad_fn(gradient):
            self.grad += np.dot(gradient, other.data)
            other.grad += np.dot(self.data, gradient)
        out.grad_fn = grad_fn

        return out
    
    def broadcast_to(self, shape):
        out = Tensor(np.broadcast_to(self.data, shape), (self,))

        def backw_op(gradient):
            self.gradient += np.sum(gradient, shape_to_axis(gradient.shape,self.data.shape), keepdims=True) \
                if tuple(gradient.shape) != tuple(self.data.shape) else gradient
        out.backw_op = backw_op
        
        return out
    
    def pad(self, arg):
        narg = tuple((p[0], s+p[0]) for s,p in zip(self.data.shape, arg))
        out = Tensor(np.pad(self.data, arg), (self,), op=self.pad)
        
        def grad_fn(gradient):
            self.grad += unpad(gradient, narg)
        out.grad_fn = grad_fn
        
        return out
    
    def shrink(self, arg):
        narg = tuple((p[0], s-p[1]) for s,p in zip(self.data.shape, arg))
        
        out = Tensor(unpad(self.data, arg), (self,), op=self.shrink)
            
        def grad_fn(gradient):
            self.grad += np.pad(gradient, narg)
        out.grad_fn = grad_fn
        
        return out
    
    def flip(self, axis):
        axis = [x if x >= 0 else x+len(self.data.shape) for x in argfix(axis)]
        arg = tuple(-1 if i in axis else 1 for i in range(len(self.data.shape)))
        
        out = Tensor(stride(self.data, arg), (self,), op=self.flip)
        
        def grad_fn(gradient):
            self.grad += stride(gradient, arg)
        out.grad_fn = grad_fn
        
        return out
    
    def flatten(self):
        input_shape = self.data.shape
        
        out = Tensor(self.data.reshape(input_shape[0],-1), (self,), op=self.flatten)
        
        def grad_fn(gradient):
            self.grad += gradient.reshape(input_shape)
        out.grad_fn = grad_fn
        
        return out
    
    def linear(self, weight, bias=None):
        # self: (batch, in_features), weight: (in_features, out_features), bias: (out_features,)
        out = Tensor(self.data @ weight.data + (bias.data if bias is not None else 0), (self, weight, bias) if bias is not None else (self, weight), op=Tensor.linear)
        def grad_fn(gradient):
            # gradient: (batch, out_features)
            # dL/dx = gradient @ w.T
            self.grad += gradient @ weight.data.T
            # dL/dw = x.T @ gradient
            weight.grad += self.data.T @ gradient
            if bias is not None:
                # dL/db = sum over batch axis
                bias.grad += gradient.sum(axis=0)
        out.grad_fn = grad_fn
        return out

    def conv2d(self, kernels, stride=1, pad=0, bias=None):
        # self: (batch_size, in_channels, input_height, input_width)
        # kernels: (out_channels, in_channels, kernel_size, kernel_size)
        out_channels, in_channels, kernel_size, _ = kernels.shape
        batch_size, in_channels2, input_height, input_width = self.shape
        assert in_channels == in_channels2, "Input and kernel in_channels must match"

        # Pad input if needed
        if pad > 0:
            x_padded = np.pad(self.data, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
        else:
            x_padded = self.data

        padded_height = x_padded.shape[2]
        padded_width = x_padded.shape[3]

        out_height = (padded_height - kernel_size) // stride + 1
        out_width = (padded_width - kernel_size) // stride + 1
        output_shape = (batch_size, out_channels, out_height, out_width)
        out = np.zeros(output_shape)

        # Perform convolution
        for k in range(batch_size):
            for i in range(out_channels):
                temp = None
                for j in range(in_channels):
                    conv = signal.correlate2d(x_padded[k, j], kernels.data[i, j], mode="valid")
                    if temp is None:
                        temp = conv
                    else:
                        temp += conv
                # Now temp is the full valid conv sum for this (k, i)
                # Apply stride
                temp_strided = temp[::stride, ::stride]
                out[k, i] = temp_strided
        
        if bias is not None:
            out += bias.data.reshape(1, out_channels, 1, 1)

        out = Tensor(out, (self, kernels, bias) if bias is not None else (self, kernels), op=self.conv2d)

        def grad_fn(gradient):
            # Gradient w.r.t. input and kernels
            self.grad = np.zeros_like(self.data)
            kernels.grad = np.zeros_like(kernels.data)
            if bias is not None:
                bias.grad = np.zeros_like(bias.data)

            # For backward, upsample the gradient if stride > 1
            grad_up = gradient
            if stride > 1:
                grad_up = np.zeros((batch_size, out_channels, padded_height - kernel_size + 1, padded_width - kernel_size + 1))
                grad_up[:, :, ::stride, ::stride] = gradient
            else:
                grad_up = gradient

            # Backprop through conv
            for k in range(batch_size):
                for i in range(out_channels):
                    for j in range(in_channels):
                        # dL/dK = x_padded * grad_up
                        kernels.grad[i, j] += signal.correlate2d(x_padded[k, j], grad_up[k, i], mode="valid")
                        # dL/dx_padded = grad_up * kernel (flipped)
                        self_grad_padded = signal.convolve2d(grad_up[k, i], kernels.data[i, j], mode="full")
                        # Crop self_grad_padded to match input (self.data) shape
                        h_start = self_grad_padded.shape[0] - self.data.shape[2]
                        w_start = self_grad_padded.shape[1] - self.data.shape[3]
                        if h_start > 0 or w_start > 0:
                            h0 = h_start // 2
                            w0 = w_start // 2
                            h1 = h0 + self.data.shape[2]
                            w1 = w0 + self.data.shape[3]
                            self_grad_padded = self_grad_padded[h0:h1, w0:w1]
                        self.grad[k, j] += self_grad_padded
            # If bias is present, sum over the batch and spatial dimensions
            if bias is not None:
                bias.grad += gradient.sum(axis=(0, 2, 3))
        out.grad_fn = grad_fn
        return out
    
    def conv_transpose2d(self, kernels, bias=None):
        """
        Transposed convolution (a.k.a. deconvolution), stride=1, padding=0.
        Input x: (N, C_in, H, W)
        Kernels: (C_in, C_out, kH, kW)
        Bias: (C_out,) or None
        Output: (N, C_out, H + kH - 1, W + kW - 1)
        """
        kT = promote_array_to_tensor(kernels)
        bT = promote_array_to_tensor(bias) if bias is not None else None

        x = self.data
        w = kT.data
        assert x.ndim == 4 and w.ndim == 4, "conv_transpose2d expects 4D input and 4D kernels"
        N, C_in, H, W = x.shape
        Cin_w, C_out, kH, kW = w.shape
        assert Cin_w == C_in, "kernels.shape[0] must equal input C_in"
        out_H, out_W = H + kH - 1, W + kW - 1

        out = np.zeros((N, C_out, out_H, out_W), dtype=x.dtype)
        # y[n, cout] = sum_cin convolve2d(x[n, cin], w[cin, cout], mode='full')
        for n in range(N):
            for cout in range(C_out):
                acc = 0
                for cin in range(C_in):
                    y = signal.convolve2d(x[n, cin], w[cin, cout], mode='full')
                    acc += y
                if bT is not None:
                    acc += bT.data[cout]
                out[n, cout] = acc

        outT = Tensor(out, (self, kT) + ((bT,) if bT is not None else ()), op=Tensor.conv_transpose2d)

        def grad_fn(grad):
            # grad: (N, C_out, out_H, out_W)
            # dL/dx[n, cin] = sum_cout correlate2d(grad[n, cout], w[cin, cout], mode='valid')
            gx = np.zeros_like(x)
            for n in range(N):
                for cin in range(C_in):
                    acc = 0
                    for cout in range(C_out):
                        g = signal.correlate2d(grad[n, cout], w[cin, cout], mode='valid')
                        acc += g
                    gx[n, cin] = acc
            self.grad += gx

            # dL/dw[cin, cout] = correlate2d(grad[n, cout], x[n, cin], mode='valid') summed over n
            gw = np.zeros_like(w)
            for cin in range(C_in):
                for cout in range(C_out):
                    acc = 0
                    for n in range(N):
                        g = signal.correlate2d(grad[n, cout], x[n, cin], mode='valid')
                        acc += g
                    gw[cin, cout] = acc
            kT.grad += gw

            if bT is not None:
                # dL/db[cout] = sum over n,h,w of grad[n, cout, h, w]
                bT.grad += grad.sum(axis=(0, 2, 3))

        outT.grad_fn = grad_fn
        return outT
    
    def maxpool2d(self, kernel_height, kernel_width, stride):
        # all params are integers
        N, C, H, W = self.data.shape
        stride = stride
        PH = kernel_height
        PW = kernel_width
        outH = int(1 + (H - PH) / stride)
        outW = int(1 + (W - PW) / stride)

        # create output tensor for pooling layer
        out = np.zeros((N, C, outH, outW))
        
        for index in range(N):
            out_col = np.zeros((C, outH*outW))
            neuron = 0
            for i in range(0, H - PH + 1, stride):
                for j in range(0, W - PW + 1, stride):
                    pool_region = self.data[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW)
                    out_col[:,neuron] = pool_region.max(axis=1)
                    neuron += 1
            out[index] = out_col.reshape(C, outH, outW)
        
        out = Tensor(out, (self,), op=self.maxpool2d)
        
        def grad_fn(gradient):
            dx = np.zeros(self.data.shape)
            
            for index in range(N):
                dout_row = gradient[index].reshape(C, outH*outW)
                neuron = 0
                for i in range(0, H-PH+1, stride):
                    for j in range(0, W-PW+1, stride):
                        pool_region = self.data[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW)
                        max_pool_indices = pool_region.argmax(axis=1)
                        dout_cur = dout_row[:,neuron]
                        neuron += 1
                        # pass gradient only through indices of max pool
                        dmax_pool = np.zeros(pool_region.shape)
                        dmax_pool[np.arange(C),max_pool_indices] = dout_cur
                        dx[index,:,i:i+PH,j:j+PW] += dmax_pool.reshape(C,PH,PW)
                        
            self.grad += dx
        out.grad_fn = grad_fn
        
        return out
            
    def l2(self):
        # computes l2 loss of a Tensor
        # can be used for regression
        out = Tensor((1/2)*np.linalg.norm(self.data, ord=2)**2, (self,), op=self.l2)
        
        def grad_fn(gradient):
            self.grad += gradient * self.data
        out.grad_fn = grad_fn
        
        return out
    
    def dropout(self, p_drop, training=True):
        if training:
            p_keep = 1 - p_drop
            binary_mask = np.random.rand(*self.data.shape) < p_keep
            result = self.data * binary_mask
            result /= p_keep # inverted dropout
        
            out = Tensor(result, (self,), op=self.dropout)
        
            def grad_fn(gradient):
                self.grad += gradient * binary_mask
            out.grad_fn = grad_fn
            
            return out
        return self
    
    def softmax(self, axis=-1):
        # computes softmax of self tensor along any axis
        x = self.data
        # subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x - x_max)
        sigma = ex / np.sum(ex, axis=axis, keepdims=True)
        out = Tensor(sigma, (self,), op=self.softmax)

        def grad_fn(gradient):
            # gradient: same shape as sigma
            # For each slice along axis, compute the Jacobian-vector product
            grad = np.zeros_like(sigma)
            # Move axis to -1 for easier iteration
            sigma_swap = np.moveaxis(sigma, axis, -1)
            grad_swap = np.moveaxis(gradient, axis, -1)
            for idx in np.ndindex(sigma_swap.shape[:-1]):
                s = sigma_swap[idx]
                g = grad_swap[idx]
                grad_swap[idx] = s * (g - np.sum(g * s)) # vjp
            # Move axis back to original position
            grad = np.moveaxis(grad_swap, -1, axis)
            self.grad += grad
        out.grad_fn = grad_fn
        
        return out
    
    def embedding(self, weight):
        # (b, t)
        # idx: integer numpy array or Tensor of shape (...,)
        idx_np = self.data.astype(np.int64)
        # Gather rows from weight
        out_data = weight.data[idx_np]
        out = Tensor(out_data, (weight,), op=self.embedding)

        def grad_fn(gradient):
            # Accumulate gradient for each index
            grad_weight = np.zeros_like(weight.data)
            # gradient has shape (..., embedding_dim)
            flat_idx = idx_np.ravel()
            flat_grad = gradient.reshape(-1, grad_weight.shape[1])
            for i, ind in enumerate(flat_idx):
                grad_weight[ind] += flat_grad[i]
            weight.grad += grad_weight
        out.grad_fn = grad_fn
        
        return out

    def split(self, size, axis=-1):
        # Split the tensor along the specified axis into chunks of the given size (only int allowed)
        assert isinstance(size, int), "Size must be an integer"
        assert size > 0, "Size must be positive"
        splits = np.split(self.data, indices_or_sections=self.data.shape[axis] // size, axis=axis)
        out_tensors = []
        idx = 0
        for chunk in splits:
            slc = [slice(None)] * self.data.ndim
            slc[axis] = slice(idx, idx + chunk.shape[axis])
            def make_grad_fn(slc_):
                def grad_fn(gradient):
                    self.grad[tuple(slc_)] += gradient
                return grad_fn
            t = Tensor(chunk, (self,), op=Tensor.split)
            t.grad_fn = make_grad_fn(slc)
            out_tensors.append(t)
            idx += chunk.shape[axis]
            
        return out_tensors


def softmax(logits, axis=-1):
    return logits.softmax(axis=axis)

def log_softmax(logits, axis=-1):
    return softmax(logits, axis=axis).log()

def l2_loss(preds, targets):
    # preds is a prediction vector
    # target contains the target values
    return (preds - targets).l2()

def mse(preds, targets):
    # preds is a prediction vector
    # target contains the target values
    return ((preds - targets)**2).mean()

def negative_log_likelihood(probs, targets):
    # binary classification
    # preds is a probability vector
    # targets is a vector of zeros and ones
    label_probs = probs * targets + (1 - probs) * (1 - targets)
    return -(label_probs.log().mean())

def categorical_cross_entropy_from_logits(logits, targets, eps=1e-8):
    # multiclass classification
    # logits vector is not a probability vector (columns dont sum to 1)
    # logits: Tensor (batch, num_classes)
    # targets: Tensor (batch, num_classes), one-hot
    logits_max = logits.max(axis=1, keepdims=True)
    exp_shifted = (logits - logits_max).exp()
    log_sum_exp = exp_shifted.sum(axis=1, keepdims=True).log()
    log_probs = logits - logits_max - log_sum_exp
    loss = -(targets * log_probs).sum(axis=1).mean()
    return loss

def sparse_categorical_crossentropy_from_logits(logits, targets, ignore_index=None, eps=1e-8):
    """
    Computes sparse categorical cross-entropy loss from logits.

    Args:
        logits of shape (batch_size, num_classes) – raw scores.
        targets of shape (batch_size,) – integer class labels in [0, num_classes).

    Returns:
        Loss (float): average negative log-probability assigned to the true classes.
    """
    # Numerical stability: subtract max logit per sample
    logits_max = logits.max(axis=1, keepdims=True)
    exp_shifted = (logits - logits_max).exp()
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    if ignore_index is not None:
        mask = (targets.data != ignore_index)
        if not np.any(mask):
            return Tensor(0.0)
        batch_indices = np.arange(logits.shape[0])[mask]
        true_classes = targets.data[mask].astype(int)
        true_probs = probs[batch_indices, true_classes]
    else:
        batch_indices = np.arange(logits.shape[0])
        true_classes = targets.data.astype(int)
        true_probs = probs[batch_indices, true_classes]
    
    # Compute mean of negative log-likelihood
    return -(true_probs + eps).log().mean()  # small epsilon to avoid log(0)