import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyad.new_core import Tensor, Transformer, sparse_categorical_crossentropy_from_logits

# ----- Torch reference model (matches shapes/structure) -----
class TorchCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TorchBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = TorchCausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x

class TorchTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Parameter(torch.zeros(block_size, n_embd))
        self.blocks = nn.ModuleList([TorchBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, ignore_index=-1):
        B, T = idx.shape
        tok = self.tok(idx)
        pos = self.pos[:T].unsqueeze(0)
        x = tok + pos
        for blk in self.blocks: x = blk(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=ignore_index)
        return logits, loss

# ----- Helper to copy weights from pyad model to torch model -----
def copy_weights(py_model, torch_model):
    # Embeddings
    torch_model.tok.weight.data.copy_(torch.tensor(py_model.token_embedding_table.W.data, dtype=torch_model.tok.weight.dtype))
    if hasattr(py_model, "position_embedding_table"):
        torch_model.pos.data.copy_(torch.tensor(py_model.position_embedding_table.W.data, dtype=torch_model.pos.dtype))
    else:
        torch_model.pos.data.zero_()

    # Blocks
    for b_idx, (py_blk, th_blk) in enumerate(zip(py_model.blocks, torch_model.blocks)):
        # LayerNorm1
        th_blk.ln1.weight.data.copy_(torch.tensor(py_blk.ln1.gamma.data))
        th_blk.ln1.bias.data.copy_(torch.tensor(py_blk.ln1.beta.data))
        # LayerNorm2
        th_blk.ln2.weight.data.copy_(torch.tensor(py_blk.ln2.gamma.data))
        th_blk.ln2.bias.data.copy_(torch.tensor(py_blk.ln2.beta.data))
        # Attention q,k,v
        th_blk.attn.q.weight.data.copy_(torch.tensor(py_blk.attn.query.W.data).T)
        th_blk.attn.k.weight.data.copy_(torch.tensor(py_blk.attn.key.W.data).T)
        th_blk.attn.v.weight.data.copy_(torch.tensor(py_blk.attn.value.W.data).T)
        # Output projection if exists
        if hasattr(py_blk.attn, "proj"):
            th_blk.attn.proj.weight.data.copy_(torch.tensor(py_blk.attn.proj.W.data).T)
        # FeedForward (assumes py_blk.ff has layers list: [LinearLayer, LinearLayer])
        # If architecture differs, adapt mapping
        py_ff0 = py_blk.ff.layers[0]
        py_ff1 = py_blk.ff.layers[2]
        th_blk.fc1.weight.data.copy_(torch.tensor(py_ff0.W.data).T)
        th_blk.fc1.bias.data.copy_(torch.tensor(py_ff0.b.data))
        th_blk.fc2.weight.data.copy_(torch.tensor(py_ff1.W.data).T)
        th_blk.fc2.bias.data.copy_(torch.tensor(py_ff1.b.data))

    # Final LayerNorm
    torch_model.lnf.weight.data.copy_(torch.tensor(py_model.ln_f.gamma.data))
    torch_model.lnf.bias.data.copy_(torch.tensor(py_model.ln_f.beta.data))
    # Head
    torch_model.head.weight.data.copy_(torch.tensor(py_model.lm_head.W.data).T)
    torch_model.head.bias.data.copy_(torch.tensor(py_model.lm_head.b.data))

# ----- Comparison run -----
def main():
    np.random.seed(123)
    torch.manual_seed(123)

    vocab_size = 32
    block_size = 16
    n_layer = 1
    n_head = 2
    n_embd = 16

    # Build pyad model (dropout assumed 0 via its config)
    cfg = type("Cfg", (), dict(vocab_size=vocab_size,
                               block_size=block_size,
                               n_layer=n_layer,
                               n_head=n_head,
                               n_embd=n_embd))()
    py_model = Transformer(cfg)

    # Build torch model
    th_model = TorchTransformer(vocab_size, block_size, n_embd, n_head, n_layer)

    # Copy weights
    copy_weights(py_model, th_model)

    # Batch
    B = 4
    X_np = np.random.randint(0, vocab_size, size=(B, block_size), dtype=np.int64)
    Y_np = X_np.copy()
    Y_np[:, :-1] = X_np[:, 1:]
    Y_np[:, -1] = -1  # ignore index

    X_py = Tensor(X_np)
    Y_py = Tensor(Y_np)
    logits_py, loss_py = py_model(X_py, Y_py)

    X_th = torch.tensor(X_np, dtype=torch.long)
    Y_th = torch.tensor(Y_np, dtype=torch.long)
    logits_th, loss_th = th_model(X_th, Y_th)

    # Compare logits
    diff_logits = np.max(np.abs(logits_py.data - logits_th.detach().numpy()))
    print("max logits diff:", diff_logits)

    # Backward
    loss_py.backward()
    loss_th.backward()

    # Collect gradients
    mismatches = []

    def chk(name, py_tensor, th_tensor, transpose=False):
        if py_tensor is None or th_tensor is None: return
        p = py_tensor.grad
        t = th_tensor.grad.detach().numpy()
        if transpose: p = p.T
        if not np.allclose(p, t, atol=1e-5):
            mismatches.append(name)

    # Embeddings
    chk("tok_emb", py_model.token_embedding_table.W, th_model.tok.weight, transpose=False)
    # Blocks
    for bi, (py_blk, th_blk) in enumerate(zip(py_model.blocks, th_model.blocks)):
        chk(f"blk{bi}.ln1.gamma", py_blk.ln1.gamma, th_blk.ln1.weight)
        chk(f"blk{bi}.ln1.beta", py_blk.ln1.beta, th_blk.ln1.bias)
        chk(f"blk{bi}.ln2.gamma", py_blk.ln2.gamma, th_blk.ln2.weight)
        chk(f"blk{bi}.ln2.beta", py_blk.ln2.beta, th_blk.ln2.bias)
        chk(f"blk{bi}.attn.q", py_blk.attn.query.W, th_blk.attn.q.weight, transpose=True)
        chk(f"blk{bi}.attn.k", py_blk.attn.key.W, th_blk.attn.k.weight, transpose=True)
        chk(f"blk{bi}.attn.v", py_blk.attn.value.W, th_blk.attn.v.weight, transpose=True)
        if hasattr(py_blk.attn, "proj"):
            chk(f"blk{bi}.attn.proj", py_blk.attn.proj.W, th_blk.attn.proj.weight, transpose=True)
        # FF
        py_ff0 = py_blk.ff.layers[0]
        py_ff1 = py_blk.ff.layers[2]
        chk(f"blk{bi}.fc1.W", py_ff0.W, th_blk.fc1.weight, transpose=True)
        chk(f"blk{bi}.fc1.b", py_ff0.b, th_blk.fc1.bias)
        chk(f"blk{bi}.fc2.W", py_ff1.W, th_blk.fc2.weight, transpose=True)
        chk(f"blk{bi}.fc2.b", py_ff1.b, th_blk.fc2.bias)

    chk("ln_f.gamma", py_model.ln_f.gamma, th_model.lnf.weight)
    chk("ln_f.beta", py_model.ln_f.beta, th_model.lnf.bias)
    chk("lm_head.W", py_model.lm_head.W, th_model.head.weight, transpose=True)
    chk("lm_head.b", py_model.lm_head.b, th_model.head.bias)

    if mismatches:
        print("Gradient mismatches:", mismatches)
    else:
        print("All gradients match within tolerance.")

if __name__ == "__main__":
    main()