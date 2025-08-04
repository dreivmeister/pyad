from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(np.int64)

# One-hot encode labels
y_onehot = np.zeros((y.size, 10), dtype=np.float32)
y_onehot[np.arange(y.size), y] = 1.0

# Split train/test
num_samples = 10000
X_train, X_test = X[:num_samples], X[num_samples:num_samples+1000]
y_train, y_test = y_onehot[:num_samples], y_onehot[num_samples:num_samples+1000]

from pyad.new_core import Tensor, MLP, categorical_cross_entropy
from pyad.optim import Adam

# Hyperparameters
batch_size = 64
epochs = 100
lr = 3e-4

mlp = MLP(784, [128, 128, 10], nonlin='relu')
optim = Adam(mlp.parameters(), lr=lr)

def get_batches(X, y, batch_size):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

for epoch in range(epochs):
    total_loss = 0
    for Xb, yb in get_batches(X_train, y_train, batch_size):
        X_tensor = Tensor(Xb)
        y_tensor = Tensor(yb)
        optim.zero_grad()
        out = mlp(X_tensor)
        loss = categorical_cross_entropy(out, y_tensor)
        loss.backward()
        optim.step()
        total_loss += loss.data
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    
# Evaluate on test set
X_tensor = Tensor(X_test)
out = mlp(X_tensor)
preds = np.argmax(out.data, axis=1)
labels = np.argmax(y_test, axis=1)
acc = np.mean(preds == labels)
print(f"Test accuracy: {acc:.4f}")