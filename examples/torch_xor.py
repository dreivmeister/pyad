import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def test_mlp_xor_pytorch():
    # XOR dataset
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

    # MLP: 2 -> 16 -> 1 with ReLU
    mlp = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='sum')

    for epoch in range(1000):
        optimizer.zero_grad()
        out = mlp(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    # Test predictions
    preds = mlp(X).detach().numpy()
    assert np.sum((preds - y.numpy()) ** 2) < 0.1, f"Test failed: {np.sum((preds - y.numpy()) ** 2)}"

    X_test = torch.tensor([[0,0],[1,1]], dtype=torch.float32)
    preds_test = mlp(X_test).detach().numpy()
    assert np.sum((preds_test - np.array([[0],[0]], dtype=np.float32)) ** 2) < 0.1, f"Test failed: {np.sum((preds_test - np.array([[0],[0]], dtype=np.float32)) ** 2)}" 

# To run the test:
if __name__ == "__main__":
    test_mlp_xor_pytorch()
    print("PyTorch XOR MLP test passed!")