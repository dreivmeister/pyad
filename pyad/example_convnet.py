import numpy as np
from new_core import Tensor, Conv2d, MaxPool2d, LinearLayer, MLP
from optim import Adam

# Example: Small ConvNet for toy image data (e.g., 4x4 grayscale images)

def small_convnet_example():
    # Fake batch of 2 grayscale images, 4x4
    bs = 2
    x = Tensor(np.random.randn(bs, 1, 4, 4))
    y = Tensor(np.array([[1, 0], [0, 1]]))  # Fake 2-class one-hot labels

    # ConvNet: Conv2d -> ReLU -> MaxPool2d -> Flatten -> LinearLayer
    conv = Conv2d(input_shape=(bs, 1, 4, 4), kernel_size=3, num_filters=2)
    pool = MaxPool2d(kernel_size=2, stride=2)
    fc = LinearLayer(2 * 1 * 1, 2)  # After conv+pool, output is (2,2,1,1)
    
    def forward(x):
        out = conv(x)           # (2,2,2,2)
        out = out.relu()        # (2,2,2,2)
        out = pool(out)         # (2,2,1,1)
        out = out.reshape((bs, -1))  # (2,2)
        logits = fc(out)        # (2,2)
        return logits
    
    optim = Adam([conv.kernels, fc.W, fc.b], alpha=0.001)

    for epoch in range(10):
        optim.zero_grad()
        # Forward pass
        logits = forward(x)

        # Loss: simple MSE to one-hot
        loss = ((logits - y) ** 2).sum()
        print('Epoch:', epoch, 'Loss:', loss.data)

        # Backward
        loss.backward()
        optim.step()

if __name__ == "__main__":
    small_convnet_example()
