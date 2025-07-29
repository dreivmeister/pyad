import numpy as np
from .core import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.gradient = None

    def parameters(self):
        return []
    
class Dense(Module):
    def __init__(self, nin, nout, bias=True, nonlin=None):
        super().__init__()
        k = np.sqrt(1/nin)
        self.W = Tensor(np.random.uniform(-k,k,(nin,nout)))
        if bias:
            self.b = Tensor(np.zeros((nout,)))
        self.bias = bias
        
        if nonlin is not None:
            self.nonlin = getattr(Tensor, nonlin)
        else:
            self.nonlin = None
        
    def __call__(self, x):
        # Ensure x is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        out = x.dot(self.W)
        
        if self.bias:
            out = out + self.b
        if self.nonlin is not None:
            out = self.nonlin(out)
        return out
    
    def parameters(self):
        if self.bias:
            return [self.W, self.b]
        return [self.W]
    
    def __repr__(self):
        return f"Dense({self.W.shape[0]}, {self.W.shape[1]}, nonlin={self.nonlin.__name__ if self.nonlin else None})"
    
class MLP(Module):
    def __init__(self, nin, nouts, nonlin='relu'):
        super().__init__()
        sizes = [nin] + nouts
        self.layers = [Dense(sizes[i], sizes[i+1], nonlin = nonlin) for i in range(len(nouts)-1)] + [Dense(sizes[-2], sizes[-1], nonlin=None)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    
class BatchNorm1D(Module):
    # input of shape (N,D)
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1):
        self.num_features = num_features
        self.gamma = Tensor(np.ones((num_features,)))
        self.beta = Tensor(np.zeros((num_features,)))
        self.momentum = momentum
        
        self.running_mean = Tensor(np.zeros((num_features,)))
        self.running_var = Tensor(np.ones((num_features,)))
        
    def __call__(self, x, training=True):
        # x is of shape (N, num_features)
        # or maybe not dont know
        # mean and var along axis=0
        if training:
            m = x.mean(axis=0)
            v = x.var(axis=0) + 1e-5
            
            # running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            
            return self.gamma * ((x - m) / v.sqrt()) + self.beta
        # testing
        return self.gamma / self.running_var * x + (self.beta - (self.gamma * self.running_mean) / self.running_var)
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
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
        
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        
    def __call__(self, x):
        # x is of shape normalized_shape
        m = x.mean(axis=self.axis_tuple)
        v = x.var(axis=self.axis_tuple) + 1e-5
        
        return ((x - m) / v.sqrt()) * self.gamma + self.beta
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_keep = 1 - p_drop
    
    def __call__(self, x, training=True):
        if training:
            binary_mask = Tensor(np.random.rand(x.shape)) < self.p_keep
            result = x * binary_mask
            return result / self.p_keep
        return x
    
    def parameters(self):
        return []
    
"""
class MaxPool2d(Module):
    def __init__(self, pool):
        self.pool = pool
        
    def __call__(self, x):
        new_shape = x.shape[:2]
        for i in [0, 1]:
            pool_w = self.pool[i]
            img_w = x.shape[i+2]
            new_shape += (img_w // pool_w, pool_w)
        result = T.reshape(x, new_shape)
        return T.max(T.max(result, axis=3), axis=4)
    
    def parameters(self):
        return []
"""
    
# only stride 1 and valid
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.in_c = in_channels # input depth
        self.out_c = out_channels # num filters
        
        kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.kernels = Tensor(np.random.rand(*kernels_shape))
        
    def __call__(self, x):
        batch_size, channels, height, width = x.shape
        out_shape = (batch_size, self.out_c, height-self.kernel_size+1, width-self.kernel_size+1)
        return Tensor.conv2d(x, self.kernels, out_shape)
    
    def parameters(self):
        return [self.kernels]
    
    
class AttentionHead(Module):
    def __init__(self, block_size, n_embd, head_size, dropout=0.2, mask=False):
        self.key = Dense(n_embd, head_size, bias=False)
        self.query = Dense(n_embd, head_size, bias=False)
        self.value = Dense(n_embd, head_size, bias=False)
        self.do_mask = mask
        if mask:
            m = np.zeros((block_size,block_size))
            m[np.triu_indices(block_size,1)] = -np.inf
            self.mask = Tensor(m)
        
        self.dropout = Dropout(dropout)
    
    def __call__(self, x):
        b, t, c = x.shape # (batch_size,block_size,n_embd)
        k = self.key(x) # (batch_size,block_size,n_embd) @ (n_embd, head_size) 
        q = self.query(x) # (B,T,C)
        wei = q @ T.transpose(k, (0,2,1)) # transpose last two dims
        if self.do_mask:
            wei = wei + self.mask
        wei = softmax(wei, axis=2) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
    
    def parameters(self):
        return [*self.key.parameters(),*self.query.parameters(),*self.value.parameters()]