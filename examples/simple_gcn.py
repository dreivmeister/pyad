import numpy as np

from pyad.new_core import Tensor, sparse_categorical_crossentropy_from_logits
from pyad.nn import GCN, LinearLayer, Module
from pyad.optim import Adam

# Toy undirected graph (4 nodes: 0-1-2-3 in a chain)
# Adjacency lists: neighbors per node
adjacency = [
    [1],      # 0 connected to 1
    [0, 2],   # 1 connected to 0 and 2
    [1, 3],   # 2 connected to 1 and 3
    [2],      # 3 connected to 2
]

# Initial node features (N, F)
np.random.seed(0)
N, F = 4, 3
X_np = np.random.randn(N, F).astype(np.float64)

# Labels (e.g., two classes)
y_idx = np.array([0, 0, 1, 1], dtype=np.int64)  # shape (N,)
num_classes = 2

class SmallGCN(Module):
    def __init__(self, num_nodes, num_features, hidden_features, num_classes):
        super().__init__()
        # Two GCN layers (separate weights)
        self.gcn1 = GCN(num_nodes, num_features, nonlin='relu')
        self.gcn2 = GCN(num_nodes, num_features, nonlin='relu')  # same feature size in this implementation
        # Linear classifier per node
        self.cls = LinearLayer(num_features, num_classes)

    def __call__(self, adjacency_list, x0: Tensor):
        # Important: GCN mutates its input via __setitem__, so pass fresh tensors or copies per forward
        h1 = self.gcn1(adjacency_list, x0)
        h2 = self.gcn2(adjacency_list, h1)
        logits = self.cls(h2)  # (N, num_classes)
        return logits

    def parameters(self):
        return self.gcn1.parameters() + self.gcn2.parameters() + self.cls.parameters()

# Train
net = SmallGCN(N, F, hidden_features=F, num_classes=num_classes)
opt = Adam(net.parameters(), lr=1e-2)

epochs = 300
for ep in range(epochs):
    # Fresh features tensor each step (GCN writes into it)
    X = Tensor(X_np.copy())
    y = Tensor(y_idx.copy())

    opt.zero_grad()
    logits = net(adjacency, X)
    loss = sparse_categorical_crossentropy_from_logits(logits, y)
    loss.backward()
    opt.step()

    if (ep + 1) % 50 == 0:
        pred = np.argmax(logits.data, axis=1)
        acc = np.mean(pred == y_idx)
        print(f"epoch {ep+1:3d} | loss {float(loss.data):.4f} | acc {acc:.2f}")

# Final predictions
X = Tensor(X_np.copy())
logits = net(adjacency, X)
pred = np.argmax(logits.data, axis=1)
print("Predictions:", pred.tolist(), "Labels:", y_idx.tolist())