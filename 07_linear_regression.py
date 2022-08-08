# 1) Design Model (input, output size, forward pass - all the different operations/layers)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare Data
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Lraining loop
num_epochs = 500

for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # empty gradients (because when we optimize the function this will SUM the gradients)
    optimizer.zero_grad()

    if epoch % 50 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch}: w = {w[0][0].item():.3f}, loss = {loss:.8f}')

# print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
