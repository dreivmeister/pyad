import numpy as np
import torch
from pyad.new_core import Tensor, Embedding

def test_embedding():
    # Parameters
    batch_size = 3
    seq_len = 4
    vocab_size = 10
    emb_dim = 6
    np.random.seed(42)
    torch.manual_seed(42)
    # Random integer indices
    idx_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    weight_np = np.random.randn(vocab_size, emb_dim)
    # PyAD
    idx = Tensor(idx_np)
    weight = Tensor(weight_np)
    out = idx.embedding(weight)
    # Torch
    idx_t = torch.tensor(idx_np, dtype=torch.long)
    weight_t = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
    emb_t = torch.nn.Embedding(vocab_size, emb_dim)
    emb_t.weight = torch.nn.Parameter(weight_t)
    out_t = emb_t(idx_t)
    # Compare forward
    assert np.allclose(out.data, out_t.detach().numpy(), atol=1e-6), f"Embedding forward mismatch: {out.data} vs {out_t.detach().numpy()}"
    # Backward
    dout = np.random.randn(*out.data.shape)
    out.backward(dout)
    out_t.backward(torch.tensor(dout, dtype=torch.float64))
    assert np.allclose(weight.grad, emb_t.weight.grad.numpy(), atol=1e-6), f"Embedding grad mismatch: {weight.grad} vs {emb_t.weight.grad.numpy()}"
    print("Embedding forward and backward test passed.")

if __name__ == "__main__":
    test_embedding()
