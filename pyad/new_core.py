import numpy as np
from math import prod
from scipy import signal
from graphviz import Digraph

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
        self.broadcast_dim = None
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
    
    def checkbroadcast(self, other):
        for n,(i,j) in enumerate(zip(self.shape, other.shape)):
            if i==j:
                continue
            if i<j:
                self.broadcast_dim = n
                break
            else:
                other.broadcast_dim = n
                break
    
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
    def uniform(low, high, shape):
        return Tensor(np.random.uniform(low,high,shape))
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
            
    def copy(self):
        new_tensor = Tensor(self.data,self.prev,self.op,self.name)
        new_tensor.grad = self.grad
        new_tensor.grad_fn = self.grad_fn
        new_tensor.broadcast_dim = self.broadcast_dim
        return new_tensor
            
    # def __getitem__(self, key):
    #     sliced_data = self.data[key]
    #     try:
    #         sliced_gradient = self.grad[key]
    #     except TypeError:
    #         sliced_gradient = 0
        
    #     sliced_tensor = Tensor(data=sliced_data, prev=(self,), op=self.__getitem__)
    #     sliced_tensor.grad = sliced_gradient
    #     sliced_tensor.broadcast_dim = self.broadcast_dim
    #     sliced_tensor.name = self.name
        
    #     def grad_fn(gradient):
    #         if isinstance(self.grad, int) and self.grad == 0:
    #             self.grad = np.zeros_like(self.data)
    #         self.grad[key] += gradient
    #     sliced_tensor.grad_fn = grad_fn
        
    #     return sliced_tensor
    
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
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), op=self.__add__)
        def grad_fn(gradient):
            self.grad += gradient #if self.broadcast_dim is None else gradient.sum(axis=self.broadcast_dim, keepdims=True)
            other.grad += gradient #if other.broadcast_dim is None else gradient.sum(axis=other.broadcast_dim, keepdims=True)
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
            if self.data.ndim == 0: # scalar * vector
                self.grad += np.dot(gradient, other.data)
            else:
                self.grad = self.grad + gradient * other.data
            
            if other.data.ndim == 0: # vector * scalar
                other.grad += np.dot(gradient, self.data)
            else:
                other.grad = other.grad + gradient * self.data
        out.grad_fn = grad_fn
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), op=self.__pow__)
        
        def grad_fn(gradient):
            self.grad = self.grad + gradient * (other * (self.data ** (other-1)))
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

    def max(self, axis=None, keepdims=False):
        m = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(m, (self,), op=self.max)
        
        def grad_fn(gradient):
            mask = np.equal(self.data, m).astype(np.float32)
            axis_sums = np.sum(mask,axis=axis,keepdims=True)
            mask = mask / axis_sums
            
            self.grad += gradient * mask 
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
        input_shape = self.data.shape
        out = Tensor(np.reshape(self.data, new_shape), (self,), op=self.reshape)
        
        def grad_fn(gradient):
            self.grad += np.reshape(gradient, input_shape)
        out.grad_fn = grad_fn
        
        return out
    
    def transpose(self, order):
        input_order = order
        out = Tensor(np.transpose(self.data, order), (self,), op=self.transpose)
        
        def grad_fn(gradient):
            self.grad += np.transpose(gradient, argsort(input_order))
        out.grad_fn = grad_fn

        return out
    
    def transpose(self, new_axis=None):
        if self.data.ndim == 2:
            out = Tensor(self.data.T, (self,), op=self.transpose)
            
            def grad_fn(gradient):
                # gradient.shape == out.shape
                self.grad += gradient.T 
            out.grad_fn = grad_fn
        # nd case
        elif new_axis is not None:
            old_shape = self.data.shape
            out = Tensor(np.transpose(self.data, new_axis), (self,), op=self.transpose)
            
            def grad_fn(gradient):
                # gradient.shape == out.shape
                self.grad += np.transpose(gradient, new_axis) 
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
            gradient_split = np.split(gradient, n, axis=axis)
            for i in range(n):
                seq[i].grad += gradient_split[i]
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
    
    def outer(self, other):
        other = promote_array_to_tensor(other)
        out = Tensor(np.outer(self.data, other.data), (self, other))

        def grad_fn(gradient):
            self.grad += np.dot(gradient, other.data)
            other.grad += np.dot(self.data, gradient)
        out.grad_fn = grad_fn

        return out
    
    # expand is basically broadcast_to
    def expand(self, new_shape):
        input_shape = self.data.shape
        out = Tensor(np.broadcast_to(self.data, new_shape), (self,), op=self.expand)
        
        def grad_fn(gradient):
            self.grad += np.sum(gradient, shape_to_axis(gradient.shape,input_shape), keepdims=True) \
                if tuple(gradient.shape) != tuple(input_shape) else gradient
        out.grad_fn = grad_fn
        
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

    def conv2d(self, kernels, output_shape):
        # only stride=1 and valid padding
        #https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py
        # self is the input image
        # other is a kernel
        # forward
        
        batch_size, out_channels, out_height, out_width = output_shape
        out = np.random.randn(*output_shape)
        
        in_channels = self.shape[1]
        #out = np.zeros_like(*output_shape)
        for k in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    out[k,i] += signal.correlate2d(self.data[k,j],kernels.data[i,j], "valid")
        
        out = Tensor(out, (self, kernels), op=self.conv2d)
        def grad_fn(gradient):
            self.grad = np.zeros_like(self.data)
            kernels.grad = np.zeros_like(kernels.data)

            for k in range(batch_size):
                for i in range(out_channels):
                    for j in range(in_channels):
                        kernels.grad[i,j] += signal.correlate2d(self.data[k,j],gradient[k,i],"valid")
                        self.grad[k,j] += signal.convolve2d(gradient[k,i],kernels.data[i,j],"full")
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
            # could also do: self.grad += out.grad * self.data
            self.grad += gradient * self.data
        out.grad_fn = grad_fn
        
        return out
    
    def dropout(self, p_drop, training=True):
        if training:
            p_keep = 1 - p_drop
            # might not work: np.random.rand(self.data.shape[1],self.data.shape[2])
            binary_mask = np.random.rand(*self.data.shape) < p_keep
            result = self.data * binary_mask
            result /= p_keep # inverted dropout
        
            out = Tensor(result, (self,), op=self.dropout)
        
            def grad_fn(gradient):
                self.grad += gradient * binary_mask
            out.grad_fn = grad_fn
            
            return out
        
        return out
    
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
        if axis == None:
            n = self.data.size
        elif axis == 0:
            n = self.data.shape[0]
        else:
            n = self.data.shape[1]
        
        def grad_fn(gradient):
            if keepdims == False:
                if axis == None or axis == 0:
                    self.grad += gradient * np.ones_like(self.data)*(1./n)
                if axis == 1:
                    self.grad += gradient.T * np.ones_like(self.data)*(1./n)
            else:
                self.grad += gradient * np.ones_like(self.data)*(1./n)
                
        out.grad_fn = grad_fn
        
        return out
    
    def var(self, axis=None, keepdims=False):
        out = Tensor(np.var(self.data, axis=axis, keepdims=keepdims), (self,), op=self.var)
        m = np.mean(self.data, axis=axis, keepdims=keepdims)
        n = self.data.size
        if axis == None:
            n = self.data.size
        elif axis == 0:
            n = self.data.shape[0]
        else:
            n = self.data.shape[1]
        
        
        def grad_fn(gradient):
            if keepdims == False:
                if axis == None or axis == 0:
                    self.grad += gradient * ((2./n)*(self.data - m))
                if axis == 1:
                    self.grad += gradient.T * ((2./n)*(self.data - np.expand_dims(m,axis=1)))
            else:
                self.grad += gradient * ((2./n)*(self.data - m))
        out.grad_fn = grad_fn
        
        return out
    
    def softmax(self, axis=1):
        # computes softmax of self tensor
        # subtract max
        self.data = self.data - np.max(self.data, axis=axis, keepdims=True)
        # Compute the softmax
        ex = np.exp(self.data)
        sigma = ex / np.sum(ex, axis=axis, keepdims=True)
        #sigma = np.reshape(sigma, (1, -1))
        print(sigma.shape)
        out = Tensor(sigma, (self,), op=self.softmax)
        
        def grad_fn(gradient):
            d_sigma = sigma * np.identity(sigma.shape[axis]) - sigma.T @ sigma
            print(d_sigma.shape)
            print(gradient.shape)
            self.grad = self.grad + gradient @ d_sigma
        out.grad_fn = grad_fn
        
        return out
    
#NN
class Module: 
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            # or: p.grad = np.zeros_like(p.data)
    
    def step(self, lr):
        # sgd update step
        for p in self.parameters():
            # p.data = p.data - lr * p.grad
            p.data -= lr * p.grad
    
    def parameters(self):
        return []
    
    
class Head(Module):
    def __init__(self, block_size, n_embd, head_size, dropout=0.2, mask=False):
        self.key = LinearLayer(n_embd, head_size, bias=False)
        self.query = LinearLayer(n_embd, head_size, bias=False)
        self.value = LinearLayer(n_embd, head_size, bias=False)
        self.do_mask = mask
        if mask:
            m = np.zeros((block_size,block_size))
            m[np.triu_indices(block_size,1)] = -np.inf
            self.mask = Tensor(m)
        
        self.dropout = Dropout(dropout)
        
    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x) # (batch_size,block_size,token_dim) @ (n_embd, head_size) 
        q = self.query(x) # (B,T,C)
        
        wei = q @ k.transpose((0,2,1)) # transpose last two dims
        if self.do_mask:
            wei += self.mask
        wei = wei.softmax(axis=2) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
    
    def parameters(self):
        return [*self.key.parameters(),*self.query.parameters(),*self.value.parameters()]

class FeedForward(Module):
    
    def __init__(self, n_embd):
        self.ll1 = LinearLayer(n_embd, 4*n_embd,nonlin='relu')
        self.ll2 = LinearLayer(4*n_embd, n_embd)
        self.drop = Dropout(0.5)
    
    def __call__(self, x):
        return self.drop(self.ll2(self.ll1(x)))
    
    def parameters(self):
        return [*self.ll1.parameters(), *self.ll2.parameters()]

class LinearLayer(Module):
    
    def __init__(self, nin, nout, bias=True, nonlin=None) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.w = Tensor(np.random.uniform(-k, k, (nout, nin)))
        
        if bias:
            self.b = Tensor(np.random.uniform(-k, k, (1,nout)))
        self.bias = bias
        self.nonlin = nonlin
    
    def __call__(self, x):
        if self.bias:
            act = x @ self.w.transpose() + self.b
        else:
            act = x @ self.w.transpose()
        return getattr(act, self.nonlin)() if self.nonlin else act
        #return act.relu() if self.nonlin else act
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        else:
            return [self.w]

class LogisticLayer(Module):
    
    def __init__(self, nfeat, bias=False) -> None:
        super().__init__()
        self.l = LinearLayer(nfeat, nfeat, bias=bias, nonlin='sigmoid')
    
    def __call__(self, x):
        return self.l(x)
    
    def parameters(self):
        # or: return [*self.l.parameters()]
        return [p for p in self.l.parameters()]
    
    
class Convolution(Module):
    # only valid only stride 1
    
    def __init__(self, input_shape, kernel_size, num_filters) -> None:
        super().__init__()
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        batch_size, in_channels, input_height, input_width = input_shape
        self.num_filters = num_filters # num of kernels
        self.input_shape = input_shape
        self.input_depth = in_channels
        self.output_shape = (batch_size, num_filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (num_filters, in_channels, kernel_size, kernel_size)
        
        self.kernels = Tensor(np.random.randn(*self.kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        #out = Tensor(np.copy(self.bias))
        return x.conv2d(self.kernels, self.output_shape)
    
    def parameters(self):
        return [self.kernels]
    
class MultiHeadAttention(Module):
    
    def __init__(self, h, d_model, T, mask=False):
        # maybe dont need T
        self.h = h # num heads
        self.d_model = d_model # embedding dim
        self.T = T # sequence length
        self.d_k = self.d_v = int(d_model/h) #dimension key,value
        self.mask = None
        if mask:
            # gen mask
            # shape might be wrong
            m = np.zeros((self.d_model,self.d_model))
            m[np.triu_indices(self.d_model,1)] = -np.inf
            self.mask = Tensor(m)
            
        
        # one list of three matrices for each head
        self.weights_heads = []
        for i in range(h):
            k = np.sqrt(1/self.d_k)
            # for each head
            #nin - number of columns
            #nout - number of rows
            
            weights_headi = [Tensor(np.random.uniform(-k, k, (self.d_model, self.d_k))),
                             Tensor(np.random.uniform(-k, k, (self.d_model, self.d_k))),
                             Tensor(np.random.uniform(-k, k, (self.d_model, self.d_v)))]
            # weights_headi = [Wi^Q,Wi^K,Wi^V]
            self.weights_heads.append(weights_headi)
        
        # output linear projection
        self.W_O = Tensor(np.random.uniform(-k, k, (int(self.h*self.d_v), self.d_model)))
        
    
    def __call__(self, Q, K, V):
        
        head_attentions = []
        for i in range(self.h):
            WQ = self.weights_heads[i][0]
            WK = self.weights_heads[i][1]
            WV = self.weights_heads[i][2]
            head_attentions.append(Attention(Q @ WQ, K @ WK, V @ WV, self.mask))
            
        concated_heads = Tensor.concatenate(head_attentions,axis=1)
        lin_proj_concated_heads = concated_heads @ self.W_O
        
        return lin_proj_concated_heads
    
    def parameters(self):
        return [self.W_O] + [mat for head in self.weights_heads for mat in head]
    
    

class MHA(Module):
    
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout=0.5, do_mask=False):
        self.heads = [Head(block_size=block_size,n_embd=n_embd,head_size=head_size,mask=do_mask) for _ in range(num_heads)]
        self.proj = LinearLayer(n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        out = Tensor.concatenate([h(x) for h in self.heads],axis=-1)
        out = self.dropout(self.proj(out))
        return out

    def parameters(self):
        return [*self.proj.parameters()] + [p for head in self.heads for p in head.parameters()]
    
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_drop = p_drop
    
    def __call__(self, x, training=True):
        return x.dropout(self.p_drop, training)
    
    def parameters(self):
        return []
    
class BatchNorm1D(Module):
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1) -> None:
        self.num_features = num_features
        self.gamma = Tensor(np.ones(self.num_features))
        self.beta = Tensor(np.zeros(self.num_features))
        self.eps = 1e-5
        self.momentum = momentum
        
        
        # not supported yet
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
        

class TemporalAffine(Module):
    """
    Inputs:
    batch_size - N
    series_length - T
    in_dim - D
    out_dim - M
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    """
    
    def __init__(self, batch_size, series_length, in_dim, out_dim):
        self.W = Tensor(np.random.rand(in_dim, out_dim))
        self.b = Tensor(np.random.rand(out_dim))
        
        self.N = batch_size
        self.T = series_length
        self.D = in_dim
        self.M = out_dim
        
    def __call__(self, x):
        # x of shape (batch_size, series_length, out_dim)
        return (x.reshape(self.N * self.T, self.D) @ self.W).reshape(self.N, self.T, self.M) + self.b
    
    def parameters(self):
        return [self.W, self.b]

class VanillaRNNBlock(Module):
    def __init__(self, N, T, D, H, h0):
        """
        - batch size - N
        - seq length - T
        - elem dim   - D
        - hidden dim - H
        """
        
        self.N = N
        self.T = T
        self.H = H
        self.prev_h = h0 # previous hidden state is h0
        self.Wx = Tensor(np.random.rand(D,H))
        self.Wh = Tensor(np.random.rand(H,H))
        self.b = Tensor(np.random.rand(H))
    
    def rnn_step(self, x):
        # x is of shape (N,D)
        return (x @ self.Wx + self.prev_h @ self.Wh + self.b).tanh()
    
    
    def __call__(self, x):
        # x is of shape (N,T,D)
        seq = []
        
        for i in range(self.T):
            step_out = self.rnn_step(x[:,i,:])
            seq.append(step_out)
            self.prev_h = step_out
        
        return Tensor.concatenate(seq).reshape((self.N,self.T,self.H))
    
    def parameters(self):
        return [self.Wx, self.Wh, self.b]
    

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
    
    
class LayerNorm(Module):
    # input of shape (N,D)
    # or other i think
    def __init__(self, normalized_shape):
        # normalized_shape is equivalent to num_features for input in form (N,num_features)
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, tuple):
            self.normalized_shape = normalized_shape
        
        # i think this is correct but might not be
        self.axis_tuple = tuple([i for i in range(1, len(self.normalized_shape)+1)])
        
        self.gamma = Tensor.ones(normalized_shape)
        self.beta = Tensor.zeros(normalized_shape)
        
    def __call__(self, x):
        # x is of shape normalized_shape
        m = Tensor.mean(x, axis=self.axis_tuple, keepdims=True)
        v = Tensor.var(x, m, axis=self.axis_tuple, keepdims=True) + 1e-5
        
        return ((x - m)/v.sqrt())*self.gamma + self.beta
        
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
#TODO: make Conv2d more flexible (stride,pad,...) and maybe faster
class Conv2d(Module):
    # only valid only stride 1
    def __init__(self, in_channels, out_channels, kernel_size):
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        #batch_size, in_channels, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        #self.num_filters = out_channels # num of kernels
        #self.input_shape = input_shape
        self.input_depth = in_channels
        self.num_filters = out_channels
        #self.output_shape = (batch_size, num_filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        
        self.kernels = Tensor.rand(self.kernels_shape)
        #self.kernels = Tensor(np.random.randn(*self.kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        #out = Tensor(np.copy(self.bias))
        output_shape = (x.shape[0],self.num_filters,x.shape[2]-self.kernel_size+1,x.shape[3]-self.kernel_size+1)
        return x.conv2d(self.kernels, output_shape)
    
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
    
class Block(Module):
    def __init__(self, block_size, n_embd, num_heads, dropout=0.5, do_mask=False):
        # block_size - context_length - length of sample
        # n_embd - embedding_dimension - d_model
        # num_heads - number of heads in MHA
        # head_size - embedding dimension in single head
        head_size = n_embd // num_heads
        self.sa = MHA(block_size,n_embd,num_heads,head_size,dropout,do_mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    def parameters(self):
        return [*self.sa.parameters(),*self.ln1.parameters(),*self.ln2.parameters(),*self.ffwd.parameters()]
        
        


def vanilla_rnn_step(x, prev_h, Wx, Wh, b):
    #https://github.com/jariasf/CS231n/blob/master/assignment3/cs231n/rnn_layers.py
    """
    all inputs are Tensors
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    """
    
    return (x @ Wx + prev_h @ Wh + b).tanh()
    
def vanilla_rnn(x, h0, Wx, Wh, b):
    """
    all inputs and outputs are tensors
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    """
    
    # hacky solution incoming
    seq = []
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    
    for i in range(T):
        prev_h = vanilla_rnn_step(x[:,i,:], prev_h, Wx, Wh, b)
        seq.append(prev_h)
    
    return Tensor.concatenate(seq).reshape((N,T,H))
    
    

    
def softmax(logits):
    # logits is a 2D Tensor
    # each row is one sample, each column is one feature
    logits -= logits.max(axis=1)
    # logits -= logits.max(axis=1,keepdims=True)
    ex = logits.exp()
    ex_sum = ex.sum(axis=1, keepdims=True)
    sigma = ex / ex_sum
    return sigma

def log_softmax(logits):
    return logits / logits.exp().sum(axis=1,keepdims=True).log()

def l2_loss(preds, targets):
    # preds is a prediction vector
    # target contains the target values
    return (preds - targets).l2()

def negative_log_likelihood(probs, targets):
    # binary classification
    # preds is a probability vector
    # targets is a vector of zeros and ones
    label_probs = probs * targets + (1 - probs) * (1 - targets)
    return -(label_probs.log().sum())

def hinge_loss(logits, targets):
    # logits is not a prob vector (columns dont sum to 1)
    num_samples = logits.data.shape[0]
    return 1./num_samples * (1. - targets * logits).relu().sum()

def logistic_prediction(inputs, targets, logistic_layer):
    preds = logistic_layer(inputs)
    return negative_log_likelihood(preds, targets)

def cross_entropy(probs, targets):
    # preds is a probability vector (each column sums to one)
    # targets is a one hot vector
    log_probs = (probs + 1e-8).log()
    return -(targets * log_probs).sum()

def Attention(Q,K,V,mask):
    # Q, K and V must be of the following shapes (Assumption)
    # Q, K, and V are Tensors
    # Q - (m, d_k)
    # K - (n, d_k), K.T - (B, d_k, n)
    # V - (n, d_v)
    # Scaled Dot-Product Attention
    # Attention(Q,K,V) = softmax((Q @ K.T) / sqrt(d_k)) @ V
    d_k = Q.shape[1]
    #d_v = V.shape[1]
    scaling_factor = np.sqrt(d_k)
    
    D = Q @ K.transpose() # (m, n)
    D = D / scaling_factor
    if mask:
        D = D + mask
    D = softmax(D)
    A = D @ V # (m, d_v)
    return A