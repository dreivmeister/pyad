from math import sqrt, log
from pyad.new_core import Tensor, log_softmax, softmax
from pyad.nn import Dropout, Embedding, LayerNorm, Module, LinearLayer

import copy


class EncoderDecoder(Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def __call__(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        emb_enc = src
        for layer in self.src_embed:
            emb_enc = layer(emb_enc)
        return self.encoder(emb_enc, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        emb_dec = tgt
        for layer in self.tgt_embed:
            emb_dec = layer(emb_dec)
        return self.decoder(emb_dec, memory, src_mask, tgt_mask)


class Generator(Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = LinearLayer(d_model, vocab)

    def __call__(self, x):
        return softmax(self.proj(x), axis=-1)
    
    
def clones(module, N):
    "Produce N identical layers."
    return [copy.deepcopy(module) for _ in range(N)] # maybe i need to return a ModuleList


class Encoder(Module):
    "Core encoder is a stack of N encoder layers"
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def __call__(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    
class SublayerConnection(Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = Dropout(dropout)

    def __call__(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(Module):
    "Encoder is made up of self-attn and feed __call__ (defined below)"
    def __init__(self, size, self_attn, feed___call__, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed___call__ = feed___call__
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def __call__(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed___call__)
    
    
class Decoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def __call__(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(Module):
    "Decoder is made of self-attn, src-attn, and feed __call__ (defined below)"

    def __init__(self, size, self_attn, src_attn, feed___call__, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed___call__ = feed___call__
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def __call__(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed___call__)
    
    
    
    
from numpy import triu_indices
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    m = Tensor.zeros(attn_shape)
    m[:, triu_indices(size, 1)] = -1e9
    return m

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = query.bmm(key.transpose((0, 1, 3, 2))) / sqrt(d_k) # (-2, -1)
    if mask is not None:
        scores = scores + mask # mask must have -1e9 at mask == 0 scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn.bmm(value), p_attn


class MultiHeadedAttention(Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(LinearLayer(d_model, d_model), 4)
        self.attn = None
        self.dropout = Dropout(p_drop=dropout)

    def __call__(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # unsqueeze in the first dimension
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]
        
        projs = []
        for lin, x_in in zip(self.linears, (query, key, value)):
            out_lin = lin(x_in)
            out_lin = out_lin.reshape((nbatches, -1, self.h, self.d_k))
            out_lin = out_lin.transpose((0, 2, 1, 3))
            projs.append(out_lin)
        query, key, value = projs
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose((0, 2, 1, 3)).reshape((nbatches, -1, self.h * self.d_k))
        
        del query
        del key
        del value
        return self.linears[-1](x)
    


class PositionwiseFeedForward(Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = LinearLayer(d_model, d_ff)
        self.w_2 = LinearLayer(d_ff, d_model)
        self.dropout = Dropout(p_drop=dropout)

    def __call__(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = Embedding(vocab, d_model)
        self.d_model = d_model

    def __call__(self, x):
        return self.lut(x) * sqrt(self.d_model)
    
    
class PositionalEncoding(Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p_drop=dropout)
        # Compute the positional encodings once in log space.
        self.pe = Tensor.zeros((max_len, d_model))
        position = Tensor([i for i in range(0, max_len)]).unsqueeze(1)
        div_term = Tensor([i for i in range(0, d_model, 2)]) * -(log(10000.0) / d_model)
        div_term = div_term.exp()
        self.pe[:, 0::2] = (position * div_term).sin()
        self.pe[:, 1::2] = (position * div_term).cos()
        self.pe = self.pe.unsqueeze(0)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
    
    
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        [Embeddings(d_model, src_vocab), c(position)],
        [Embeddings(d_model, tgt_vocab), c(position)],
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.ndim() > 1:
            p.xavier_uniform()
    return model



from numpy import argmax
def inference_test():
    test_model = make_model(11, 11, 2)
    src = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = Tensor.ones((1, 1, 10))

    memory = test_model.encode(src, src_mask)
    ys = Tensor.zeros((1, 1))

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.shape[1])
        )
        prob = test_model.generator(out[:, -1])
        # next_word = prob.max(axis=1)
        # next_word = int(next_word.data[0])  # ensure a Python int/scalar
        next_word = int(argmax(prob.data, axis=1)[0])
        # create a (1,1) Tensor filled with next_word and same dtype as src.data
        next_t = Tensor([[next_word]])
        ys = Tensor.concatenate([ys, next_t], axis=1)

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(3):
        inference_test()
        
        
run_tests()