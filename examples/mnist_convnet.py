from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(np.int64)

# One-hot encode labels
y_onehot = np.zeros((y.size, 10), dtype=np.float32)
y_onehot[np.arange(y.size), y] = 1.0

# Reshape X to (N, 1, 28, 28) for ConvNet
X = X.reshape(-1, 1, 28, 28)

# Split train/test
num_samples = 1000
num_test_samples = 100
X_train, X_test = X[:num_samples], X[num_samples:num_samples+num_test_samples]
y_train, y_test = y_onehot[:num_samples], y_onehot[num_samples:num_samples+num_test_samples]

from pyad.new_core import Tensor, Conv2d, MaxPool2d, LayerNorm, LinearLayer, categorical_cross_entropy
from pyad.optim import Adam

# Hyperparameters
batch_size = 128
epochs = 10
lr = 3e-4

# Define a simple ConvNet
class SimpleConvNet:
    def __init__(self):
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=5) # (8, 24, 24)
        self.ln1 = LayerNorm((8, 24, 24))
        self.pool1 = MaxPool2d(kernel_size=2, stride=2) # (8, 12, 12)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=5) # (16, 8, 8)
        self.ln2 = LayerNorm((16, 8, 8))
        self.pool2 = MaxPool2d(kernel_size=2, stride=2) # (16, 4, 4)
        self.fc = LinearLayer(16 * 4 * 4, 10)  # After conv+pool: (batch, 16, 4, 4)
    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters() + self.fc.parameters() + self.ln1.parameters() + self.ln2.parameters()
    def __call__(self, x):
        out = self.conv1(x)
        out = out.relu()
        out = self.ln1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = out.relu()
        out = self.ln2(out)
        out = self.pool2(out)
        out = out.reshape((x.data.shape[0], -1))
        out = self.fc(out)
        return out

net = SimpleConvNet()
optim = Adam(net.parameters(), lr=lr)

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
        out = net(X_tensor)
        loss = categorical_cross_entropy(out, y_tensor)
        loss.backward()
        optim.step()
        total_loss += loss.data
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate on test set
X_tensor = Tensor(X_test)
out = net(X_tensor)
preds = np.argmax(out.data, axis=1)
labels = np.argmax(y_test, axis=1)
acc = np.mean(preds == labels)
print(f"Test accuracy: {acc:.4f}")