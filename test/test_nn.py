import numpy as np
from pyad.core import Tensor
from pyad.nn import MLP
from pyad.optim import Adam

def test_mlp_training():
    # Simple dataset: XOR
    X = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
    y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float32))
    print(X.shape, y.shape)

    # Create MLP: 2 inputs, 2 hidden, 1 output
    mlp = MLP(2, [2, 1], nonlin='tanh')
    optim = Adam(mlp.parameters())
    
    #for epoch in range(200):
    total_loss = 0
    mlp.zero_grad()
    out = mlp(X)
    print(out.shape)
    loss = ((out - y) ** 2).sum()
    loss.backward()
    optim.step()
    total_loss += loss.data
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X_tensor)}")

    # Test predictions
    preds = [mlp(xi).data for xi in X_tensor]
    assert all((p > 0.5) == t for p, t in zip(preds, y.flatten()))
    
    
test_mlp_training()