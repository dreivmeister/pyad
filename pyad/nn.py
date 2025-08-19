import numpy as np
from pyad.new_core import Tensor, sparse_categorical_crossentropy_from_logits


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
        self.W = Tensor.uniform(-k, k, (nin, nout))
        self.bias = bias
        if bias:
            self.b = Tensor.uniform(-k, k, (nout,))
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
        params = [self.kernels]
        if self.bias is not None:
            params.append(self.bias)
        return params

class ConvTranspose2d(Module):
    # stride=1, padding=0 (output size grows by k-1 per spatial dim)
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        # kernels: (C_in, C_out, kH, kW) for transposed conv
        self.kernels = Tensor.randn((in_channels, out_channels, kH, kW))
        self.bias = Tensor.zeros((out_channels,)) if bias else None

    def __call__(self, x):
        return x.conv_transpose2d(self.kernels, bias=self.bias)

    def parameters(self):
        # mirror Conv2d style; include bias if you want it trained
        params = [self.kernels]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
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
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_drop = p_drop
    
    def __call__(self, x, training=True):
        return x.dropout(self.p_drop, training)
    
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

class CausalMultiHeadSelfAttention(Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head # head size
        self.scale = 1.0 / np.sqrt(self.head_size)
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
        self.scale = 1.0 / np.sqrt(self.head_size)
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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.xh_to_h = LinearLayer(input_size + hidden_size, hidden_size)

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
    

class LSTMCell(Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_i = LinearLayer(input_size + hidden_size, hidden_size)
        self.xh_to_f = LinearLayer(input_size + hidden_size, hidden_size)
        self.xh_to_g = LinearLayer(input_size + hidden_size, hidden_size)
        self.xh_to_o = LinearLayer(input_size + hidden_size, hidden_size)

    def __call__(self, xt, hprev, cprev):
        # xt - (b, input_size)
        # hprev - (b, hidden_size)
        # cprev - (b, hidden_size)
        
        xh = Tensor.concatenate([xt, hprev], axis=1) # (b, input_size + hidden_size)
        i = self.xh_to_i(xh).sigmoid() # input
        f = self.xh_to_f(xh).sigmoid() # forget
        g = self.xh_to_g(xh).tanh()
        o = self.xh_to_o(xh).sigmoid()

        c = f * cprev + i * g # update cell state
        ht = o * c.tanh() # update hidden state
        return ht, c
    
    def parameters(self):
        return [*self.xh_to_i.parameters(), *self.xh_to_f.parameters(), *self.xh_to_g.parameters(), *self.xh_to_o.parameters()]
    
    
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # First layer takes input_size, subsequent layers take hidden_size as input
        self.layers = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        self.out_proj = LinearLayer(hidden_size, input_size)

    def __call__(self, x, h0=None, c0=None):
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

        # Prepare initial cell states
        if c0 is None:
            c_list = [Tensor.zeros((B, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            # c0 can be Tensor of shape (num_layers, B, H) or list of Tensors [(B,H), ...]
            if isinstance(c0, Tensor):
                assert c0.ndim == 3 and c0.shape[0] == self.num_layers and c0.shape[1] == B and c0.shape[2] == self.hidden_size
                c_list = [c0[i, :, :] for i in range(self.num_layers)]
            else:
                assert isinstance(c0, list) and len(c0) == self.num_layers
                c_list = c0

        out = x
        h_last = []
        c_last = []
        for layer_idx, cell in enumerate(self.layers):
            hprev = h_list[layer_idx]
            cprev = c_list[layer_idx]
            ts = []
            for t in range(T):
                xt = out[:, t, :]   # (B, in_features), x or last hidden states from last layer
                hprev, cprev = cell(xt, hprev, cprev)  # (B, hidden_size)
                ts.append(hprev)
            # (B, T, hidden_size)
            out = Tensor.stack(ts, axis=1)
            h_last.append(hprev)
            c_last.append(cprev)

        # h_n: (num_layers, B, hidden_size), last hidden states from each layer
        h_n = Tensor.stack(h_last, axis=0)
        c_n = Tensor.stack(c_last, axis=0) # same for cell states
        return out, h_n, c_n

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    
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
    
    
class RNN(Module):
    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.nembd2 = config.n_embd2 # hidden state dimension
        #self.start = Tensor.zeros(1, config.n_embd2) # the starting hidden state
        self.wte = Embedding(config.vocab_size, config.n_embd) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(config.n_embd, config.n_embd2)
        elif cell_type == 'gru':
            self.cell = GRUCell(config.n_embd, config.n_embd2)
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
    

class GCN(Module):
    def __init__(self, num_nodes, num_features, nonlin=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.W = LinearLayer(num_features, num_features, bias=False)
        self.B = LinearLayer(num_features, num_features, bias=False)
        self.nonlin = nonlin

    def __call__(self, adjacency_list, initial_node_features):
        # [[0, 1], [0, 2], [1, 2], []]
        embeddings = initial_node_features
        for node in range(self.num_nodes):
            neighbors = adjacency_list[node]
            if len(neighbors) == 0:
                act = self.B(embeddings[node])
                embeddings[node] = getattr(act, self.nonlin)() if self.nonlin else act
            else:
                neighbor_embeddings = embeddings[neighbors] # (num_neighbors, num_features)
                neighbor_mean = neighbor_embeddings.mean(axis=0) # (num_features,) mean neighbor embeddings
                act = self.W(neighbor_mean) + self.B(embeddings[node])
                embeddings[node] = getattr(act, self.nonlin)() if self.nonlin else act
        return embeddings

    def parameters(self):
        return [*self.W.parameters(), *self.B.parameters()]
    
    
class GAT(Module):
    def __init__(self, num_nodes, num_features, attention_mechanism, nonlin=None):
        self.num_nodes = num_nodes
        self.W = LinearLayer(num_features, num_features, bias=False)
        self.attention_mechanism = attention_mechanism
        self.nonlin = nonlin
        
    def __call__(self, adjacency_list, initial_node_features):
        embeddings = initial_node_features
        for node in range(self.num_nodes):
            neighbors = adjacency_list[node]
            if len(neighbors) == 0:
                self_attention = self.attention_mechanism(embeddings[node], embeddings[node])
                new_embedding = self.W(self_attention * embeddings[node])
            else:
                neighbor_embeddings = embeddings[neighbors]
                # attention of node to each of its neighbors
                self_attention = self.attention_mechanism(embeddings[node], embeddings[node])
                attention_weights = self.attention_mechanism(embeddings[node], neighbor_embeddings)
                attention_weights = attention_weights.reshape((len(neighbors), -1))  # (num_neighbors, 1)
                new_embedding = self.W((neighbor_embeddings.transpose()).dot(attention_weights) + self_attention * embeddings[node])
            if self.nonlin:
                new_embedding = getattr(new_embedding, self.nonlin)()
            embeddings[node] = new_embedding
        return embeddings

    def parameters(self):
        return [*self.W.parameters(), *self.attention_mechanism.parameters()]