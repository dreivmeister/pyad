from core import Tensor

# Forward test
x = Tensor(3.0)
y = Tensor([4.0, 5.0, 6.0])
z = x * y   # self * other

print(z.data) # [12.0, 15.0, 18.0]
# Backward test
z.sum().backward()

print(x.gradient.data)
print(y.gradient.data)