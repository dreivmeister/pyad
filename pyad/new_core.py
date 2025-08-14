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
        #self.broadcast_dim = None
        self.name = name
        
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
            
    def copy(self):
        new_tensor = Tensor(self.data,self.prev,self.op,self.name)
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
    
    # def __setitem__(self, key, value):
    #     # value is a tensor
    #     self.data[key] = value.data
    #     try:
    #         self.grad[key] = value.grad
    #     except TypeError:
    #         self.grad = value.grad
        
    #     # or also:
    #     #value.prev.add(value)
    #     self.prev = (value,)
    #     self.op = self.__setitem__
        
    #     def grad_fn(gradient):
    #         if isinstance(value.grad, int) and value.grad == 0:
    #             value.grad = np.zeros_like(value.data)
    #         value.grad += gradient[key]
    #     self.grad_fn = grad_fn
        
    #     self.broadcast_dim = value.broadcast_dim
    #     self.name = value.name
    
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
        # torch-like expand: -1 keeps size, can add leading dims, only expand dims of size 1
        orig = self.data.shape
        if len(sizes) < len(orig):
            raise AssertionError("expand: target dims must be >= input dims")
        padded = (1,) * (len(sizes) - len(orig)) + orig

        target = []
        for s_in, s_out in zip(padded, sizes):
            if s_out == -1:
                target.append(s_in)
            else:
                assert s_out == s_in or s_in == 1, f"expand: cannot expand {s_in} -> {s_out}"
                target.append(int(s_out))

        out_data = np.broadcast_to(self.data.reshape(padded), tuple(target))
        out = Tensor(out_data, (self,), op=Tensor.expand)

        # axes where broadcasting happened (s_in==1 and expanded >1), including added leading dims
        reduce_axes = tuple(i for i, (s_in, s_out) in enumerate(zip(padded, target)) if s_in == 1 and s_out > 1)

        def grad_fn(grad):
            g = grad
            if reduce_axes:
                g = g.sum(axis=reduce_axes, keepdims=True)
            # reshape back to original rank
            g = g.reshape(padded).reshape(orig)
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

#NN
class Module: 
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    
    def parameters(self):
        return []

class LinearLayer(Module):
    def __init__(self, nin, nout, bias=True, nonlin=None) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.W = Tensor(np.random.uniform(-k, k, (nin, nout)))
        self.bias = bias
        if bias:
            self.b = Tensor(np.random.uniform(-k, k, (nout,)))
        else:
            self.b = None
        self.nonlin = nonlin

    def __call__(self, x):
        
        bs = None
        if x.ndim == 3:
            # If x is of shape (batch_size, seq_len, nin), reshape to (batch_size*seq_len, nin)
            bs = x.shape[0]
            x = x.reshape((-1, x.shape[-1]))
        
        act = x.linear(self.W, self.b) if self.bias else x.linear(self.W)
        act = getattr(act, self.nonlin)() if self.nonlin else act
        
        if bs is not None:
            # Reshape back to (batch_size, seq_len, nout)
            act = act.reshape((bs, -1, act.shape[-1]))
            
        return act

    def parameters(self):
        if self.bias:
            return [self.W, self.b]
        else:
            return [self.W]
        
class Embedding(Module):
    """
    Embedding layer: maps integer indices to embedding vectors.
    Like torch.nn.Embedding(num_embeddings, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) / np.sqrt(embedding_dim))

    def __call__(self, idx):
        # idx is a Tensor of shape (..., num_indices)
        return idx.embedding(self.weight)

    def parameters(self):
        return [self.weight]
        
        
class MLP(Module):
    def __init__(self, nin, nouts, nonlin='relu') -> None:
        super().__init__()
        # nin is an integer
        # nouts is a list of integers
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            if i != len(nouts)-1:
                self.layers.append(LinearLayer(sizes[i],sizes[i+1],nonlin=nonlin))
            else:
                self.layers.append(LinearLayer(sizes[i],sizes[i+1],nonlin=False))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class Conv2d(Module):
    # only valid only stride 1
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        # out_channels is num_filters
        #self.normalized_shape = (out_channels, input_height - kernel_size + 1, input_width - kernel_size + 1)
        kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None
        self.kernels = Tensor(np.random.randn(*kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        # output is a Tensor of shape (batch_size, num_filters, height - kernel_size + 1, width - kernel_size + 1)
        return x.conv2d(self.kernels, bias=self.bias)
    
    def parameters(self):
        return [self.kernels]
    
class MaxPool2d(Module):
    
    def __init__(self, kernel_size, stride):
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        
        self.stride = stride
        
    def __call__(self, x):
        # x is (N,C,H,W)
        return x.maxpool2d(self.kernel_height, self.kernel_width, self.stride)

    def parameters(self):
        return []

class FeedForward(Module):
    def __init__(self, n_embd):
        self.ll1 = LinearLayer(n_embd, 4*n_embd, nonlin='gelu')
        self.ll2 = LinearLayer(4*n_embd, n_embd)
        #self.drop = Dropout(0.5)
    
    def __call__(self, x):
        return self.ll2(self.ll1(x))
        #return self.drop(self.ll2(self.ll1(x)))
    
    def parameters(self):
        return [*self.ll1.parameters(), *self.ll2.parameters()]
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_drop = p_drop
    
    def __call__(self, x, training=True):
        return x.dropout(self.p_drop, training)
    
    def parameters(self):
        return []

from math import sqrt
class CausalMultiHeadSelfAttention(Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head # head size
        self.scale = 1.0 / sqrt(self.head_size)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LinearLayer(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = LinearLayer(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        m = np.zeros((block_size,block_size))
        m[np.triu_indices(block_size,1)] = float('-inf')
        m = np.expand_dims(m, axis=(0,1))
        self.bias = m
        self.n_head = n_head
        self.n_embd = n_embd

    def __call__(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, axis=2)
        k = k.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        q = q.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        v = v.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = q.bmm(k.transpose((0, 1, 3, 2))) * self.scale
        att = att + self.bias[:,:,:T,:T]
        att = att.softmax(axis=-1)
        y = att.bmm(v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # (B, T, nh, hs) -> (B, T, nh * hs = C)
        y = y.transpose((0, 2, 1, 3)).reshape((B, T, C)) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
    def parameters(self):
        return [*self.c_attn.parameters(), *self.c_proj.parameters()]
    
class MultiHeadSelfAttention(Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head # head size
        self.scale = 1.0 / sqrt(self.head_size)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LinearLayer(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = LinearLayer(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd

    def __call__(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, axis=2)
        k = k.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        q = q.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        v = v.reshape((B, T, self.n_head, self.head_size)).transpose((0, 2, 1, 3)) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = q.bmm(k.transpose((0, 1, 3, 2))) * self.scale
        att = att.softmax(axis=-1)
        y = att.bmm(v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # (B, T, nh, hs) -> (B, T, nh * hs = C)
        y = y.transpose((0, 2, 1, 3)).reshape((B, T, C)) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
    def parameters(self):
        return [*self.c_attn.parameters(), *self.c_proj.parameters()]
    
class Block(Module):
    def __init__(self, block_size, n_embd, num_heads):
        # block_size - context_length - length of sample
        # n_embd - embedding_dimension - d_model
        # num_heads - number of heads in MHA
        # head_size - embedding dimension in single head
        self.attn = CausalMultiHeadSelfAttention(n_embd, num_heads, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    def parameters(self):
        return [*self.attn.parameters(),*self.ln1.parameters(),*self.ln2.parameters(),*self.ffwd.parameters()]

class Transformer(Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config

        self.blocks = [Block(config.block_size, config.n_embd, config.n_head, do_mask=True) for _ in range(config.n_layer)]
        self.wte = Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = Embedding(config.block_size, config.n_embd)  # position embeddingss
        self.ln_f = LayerNorm(config.n_embd)  # final layer norm

        # language model head, maps to vocab size
        self.lm_head = LinearLayer(config.n_embd, config.vocab_size, bias=False, nonlin=None)

    def get_block_size(self):
        return self.block_size

    def __call__(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = Tensor([[i for i in range(t)]]) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd) (b, t) @ (vocab_size, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # logits of shape (b, t, vocab_size)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = sparse_categorical_crossentropy_from_logits(
                logits.reshape((-1, logits.shape[-1])), targets.reshape((-1,)), ignore_index=-1)
            
        return logits, loss
    
    def parameters(self):
        return [*self.wte.parameters(), *self.wpe.parameters(), *self.ln_f.parameters(), *self.lm_head.parameters()] + [p for block in self.blocks for p in block.parameters()]

class BatchNorm1D(Module):
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1) -> None:
        self.num_features = num_features
        self.gamma = Tensor(np.ones(self.num_features))
        self.beta = Tensor(np.zeros(self.num_features))
        self.eps = 1e-5
        self.momentum = momentum
        
        self.running_mean = Tensor(np.zeros(self.num_features))
        self.running_var = Tensor(np.ones(self.num_features))
        
    def __call__(self, x, training=True):
        # x is of shape (N, num_features)
        # or maybe not dont know
        # mean and var along axis=0
        if training:
            m = x.mean(axis=0, keepdims=True)
            v = x.var(axis=0, keepdims=True) + self.eps
            
            # running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            
            
            return self.gamma*((x - m)/v.sqrt()) + self.beta
        # testing
        return self.gamma/self.running_var * x + (self.beta - (self.gamma*self.running_mean)/self.running_var)
    
    def parameters(self):
        return [self.gamma, self.beta]    
    
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.gamma = Tensor.ones(self.normalized_shape)
        self.beta  = Tensor.zeros(self.normalized_shape)

    def __call__(self, x):
        # Normalize over the LAST len(normalized_shape) dims
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        m = x.mean(axis=axes, keepdims=True)
        v = x.var(axis=axes, keepdims=True)  # already +eps in division below
        return ((x - m) / (v + self.eps).sqrt()) * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]
    
class RNNCell(Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = LinearLayer(config.n_embd + config.n_embd2, config.n_embd2)

    def __call__(self, xt, hprev):
        xh = Tensor.concatenate([xt, hprev], axis=1)
        ht = self.xh_to_h(xh).tanh()
        return ht

    def parameters(self):
        return [*self.xh_to_h.parameters()]
    
    
class GRUCell(Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = LinearLayer(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = LinearLayer(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = LinearLayer(config.n_embd + config.n_embd2, config.n_embd2)

    def __call__(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = Tensor.concatenate([xt, hprev], axis=1)
        r = self.xh_to_r(xh).sigmoid()
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = Tensor.concatenate([xt, hprev_reset], axis=1)
        hbar = self.xh_to_hbar(xhr).tanh()
        # calculate the switch gate that determines if each channel should be updated at all
        z = self.xh_to_z(xh).sigmoid()
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        
        return ht
    
    def parameters(self):
        return [*self.xh_to_z.parameters(), *self.xh_to_r.parameters(), *self.xh_to_hbar.parameters()]
    
    
class RNN(Module):
    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.nembd2 = config.n_embd2 # hidden state dimension
        #self.start = Tensor.zeros(1, config.n_embd2) # the starting hidden state
        self.wte = Embedding(config.vocab_size, config.n_embd) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = LinearLayer(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def __call__(self, idx, targets=None):
        b, t = idx.shape

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = Tensor.zeros((b, self.nembd2))
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2) 
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = Tensor.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = sparse_categorical_crossentropy_from_logits(
                logits.reshape((-1, logits.shape[-1])), targets.reshape((-1,)), ignore_index=-1)

        return logits, loss
    
    def parameters(self):
        return [*self.wte.parameters(), *self.cell.parameters(), *self.lm_head.parameters()]



class GRUCell(Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = LinearLayer(input_size + hidden_size, hidden_size)
        self.xh_to_r = LinearLayer(input_size + hidden_size, hidden_size)
        self.xh_to_hbar = LinearLayer(input_size + hidden_size, hidden_size)

    def __call__(self, xt, hprev):
        # xt - (b, input_size)
        # hprev - (b, hidden_size)
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = Tensor.concatenate([xt, hprev], axis=1) # (b, input_size + hidden_size)
        r = self.xh_to_r(xh).sigmoid()
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = Tensor.concatenate([xt, hprev_reset], axis=1)
        hbar = self.xh_to_hbar(xhr).tanh()
        # calculate the switch gate that determines if each channel should be updated at all
        z = self.xh_to_z(xh).sigmoid()
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        
        return ht
    
    def parameters(self):
        return [*self.xh_to_z.parameters(), *self.xh_to_r.parameters(), *self.xh_to_hbar.parameters()]
    
    
class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # First layer takes input_size, subsequent layers take hidden_size as input
        self.layers = [GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        self.out_proj = LinearLayer(hidden_size, input_size)

    def __call__(self, x, h0=None):
        # x: (B, T, input_size)  (batch-first)
        B, T, _ = x.shape

        # Prepare initial hidden states
        if h0 is None:
            h_list = [Tensor.zeros((B, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            # h0 can be Tensor of shape (num_layers, B, H) or list of Tensors [(B,H), ...]
            if isinstance(h0, Tensor):
                assert h0.ndim == 3 and h0.shape[0] == self.num_layers and h0.shape[1] == B and h0.shape[2] == self.hidden_size
                h_list = [h0[i, :, :] for i in range(self.num_layers)]
            else:
                assert isinstance(h0, list) and len(h0) == self.num_layers
                h_list = h0

        out = x
        h_last = []
        for layer_idx, cell in enumerate(self.layers):
            hprev = h_list[layer_idx]
            ts = []
            for t in range(T):
                xt = out[:, t, :]   # (B, in_features)
                hprev = cell(xt, hprev)  # (B, hidden_size)
                ts.append(hprev)
            # (B, T, hidden_size)
            out = Tensor.stack(ts, axis=1)
            h_last.append(hprev)

        # h_n: (num_layers, B, hidden_size)
        h_n = Tensor.stack(h_last, axis=0)
        return out, h_n

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


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