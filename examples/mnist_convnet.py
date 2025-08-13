from sklearn.datasets import fetch_openml
import numpy as np

from pyad.new_core import Tensor, Conv2d, MaxPool2d, LinearLayer, categorical_cross_entropy_from_logits, Module
from pyad.optim import Adam

# --------------------------
# Data
# --------------------------
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(np.int64)

# One-hot
y_onehot = np.zeros((y.size, 10), dtype=np.float32)
y_onehot[np.arange(y.size), y] = 1.0

# NCHW
X = X.reshape(-1, 1, 28, 28)

# Split train/test
num_samples = 10
num_test_samples = 2
X_train, X_test = X[:num_samples], X[num_samples:num_samples+num_test_samples]
y_train, y_test = y_onehot[:num_samples], y_onehot[num_samples:num_samples+num_test_samples]

# --------------------------
# Model
# --------------------------
class ResidualBlock(Module):
    def __init__(self, in_ch, out_ch):
        # Two 3x3 convs (valid), ReLU in between. Projection on skip if channels change.
        self.conv1 = Conv2d(in_ch, out_ch, kernel_size=3)  # shrinks H,W by 2
        self.conv2 = Conv2d(out_ch, out_ch, kernel_size=3) # shrinks H,W by another 2
        self.proj = None
        if in_ch != out_ch:
            self.proj = Conv2d(in_ch, out_ch, kernel_size=1)  # keeps H,W

    def _center_crop(self, x, target_h, target_w):
        # x: (N,C,H,W) -> crop center to (target_h, target_w)
        H, W = x.data.shape[2], x.data.shape[3]
        dh = H - target_h
        dw = W - target_w
        assert dh >= 0 and dw >= 0
        top = dh // 2
        left = dw // 2
        return x[:, :, top:top+target_h, left:left+target_w]

    def __call__(self, x):
        y = self.conv1(x).relu()
        y = self.conv2(y)
        # Prepare skip
        skip = x
        if self.proj is not None:
            skip = self.proj(skip)
        # Match spatial dims by center-cropping skip
        _, _, th, tw = y.data.shape
        skip = self._center_crop(skip, th, tw)
        # Residual add + ReLU
        return (y + skip).relu()

    def parameters(self):
        params = [*self.conv1.parameters(), *self.conv2.parameters()]
        if self.proj is not None:
            params = params + self.proj.parameters()
        return params

class SmallResNet(Module):
    def __init__(self):
        # Stem: (1,28,28) -> (16,26,26)
        self.stem = Conv2d(1, 16, kernel_size=3)

        # Stage 1: (16,26,26) -> Block -> (16,22,22) -> Pool -> (16,11,11)
        self.block1 = ResidualBlock(16, 16)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: (16,11,11) -> Block -> (32,7,7) -> Pool -> (32,3,3)
        self.block2 = ResidualBlock(16, 32)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: (32,3,3) is too small for a full 2x conv block.
        # Instead, build Block at 7x7 before pooling:
        # So reorder: apply block2 at (16,11,11) to get (32,7,7),
        # then block3 to (64,3,3), then pool to (64,1,1)
        self.block3 = ResidualBlock(32, 64)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)  # (64,3,3) -> (64,1,1)

        self.fc = LinearLayer(64, 10)

    def __call__(self, x):
        out = self.stem(x).relu()             # (N,16,26,26)
        out = self.block1(out)                # (N,16,22,22)
        out = self.pool1(out)                 # (N,16,11,11)
        out = self.block2(out)                # (N,32,7,7)
        out = self.block3(out)                # (N,64,3,3)
        out = self.pool3(out)                 # (N,64,1,1)
        out = out.reshape((x.data.shape[0], -1))  # (N,64)
        out = self.fc(out)                    # (N,10)
        return out

    def parameters(self):
        return (
            self.stem.parameters()
            + self.block1.parameters()
            + self.block2.parameters()
            + self.block3.parameters()
            + self.fc.parameters()
        )

# --------------------------
# Training
# --------------------------
def get_batches(X, y, batch_size):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

batch_size = 128
epochs = 10
lr = 3e-4

net = SmallResNet()
optim = Adam(net.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0.0
    for Xb, Yb in get_batches(X_train, y_train, batch_size):
        Xb_t = Tensor(Xb)
        Yb_t = Tensor(Yb)

        optim.zero_grad()
        logits = net(Xb_t)
        loss = categorical_cross_entropy_from_logits(logits, Yb_t)  # assumes one-hot y
        loss.backward()
        optim.step()

        total_loss += float(loss.data)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --------------------------
# Eval
# --------------------------
X_test_t = Tensor(X_test)
logits = net(X_test_t)
preds = np.argmax(logits.data, axis=1)
labels = np.argmax(y_test, axis=1)
acc = np.mean(preds == labels)
print(f"Test accuracy: {acc:.4f}")