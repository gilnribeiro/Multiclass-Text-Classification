# 1) Design Model (input, output size, forward pass - all the different operations/layers)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

from logging import lastResort
import torch
import torch.nn as nn
# f = w * x

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'N_samples: {n_samples}, N_features: {n_features}')

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)

# Use a class as a wrapper


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)


# Manual model prediction
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#     return w * x

# loss = MSE

# Manually Defined
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# Manually gradient
# MSE = 1/N * (w*x -y)**2
# dJ/dw = 1/N * 2x (wx - y)

# def gradient(x, y, y_predicted):
#     return torch.dot(2*x, y_predicted-y)/len(x)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 500
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = foward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    # dw = gradient(X, Y, y_pred)
    l.backward()  # calculate gradient dl/dw

    # Manually update weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # Manually zero gradients
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 50 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
