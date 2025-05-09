import torch
import torch.nn as nn
import torch.optim as optim

# XOR data
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# Predictive coding layer
class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.W.t()) + self.b)

# Predictive coding network
class PCNet(nn.Module):
    def __init__(self, dims, device, batch_size=4):
        super().__init__()
        self.device = device
        self.dims = dims
        self.batch_size = batch_size

        # Layers
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.to(device)

        # Activations (created once, never overwritten)
        self.activations = []
        for dim in dims:
            act = torch.zeros((batch_size, dim), device=device, requires_grad=True)
            self.activations.append(act)

        # Optimizers
        self.internal_activations = self.activations[1:-1]
        self.clamped_in_optimizer = optim.Adam(self.activations[1:], lr=0.1)
        self.clamped_both_optimizer = optim.Adam(self.internal_activations, lr=0.1)
        self.learning_optimizer = optim.Adam(self.parameters(), lr=0.01)

    def compute_errors_and_loss(self):
        self.errors = []
        self.loss = 0.0
        for i, layer in enumerate(self.layers):
            pred = layer(self.activations[i])
            err = self.activations[i + 1] - pred
            self.errors.append(err)
            self.loss += (err ** 2).sum()

    def stabilize(self, optimizer, steps):
        for _ in range(steps):
            optimizer.zero_grad()
            self.compute_errors_and_loss()
            self.loss.backward(retain_graph=True)
            optimizer.step()

    def train(self, X, Y, epochs=3000):
        X, Y = X.to(self.device), Y.to(self.device)

        for epoch in range(epochs):
            # Clamp input/output without overwriting
            with torch.no_grad():
                self.activations[0].data.copy_(X)
                self.activations[-1].data.copy_(Y)
                
            self.activations[0].requires_grad_()
            self.activations[-1].requires_grad_()

            # Inference
            self.stabilize(self.clamped_both_optimizer, steps=30)

            # Learning
            self.compute_errors_and_loss()
            self.learning_optimizer.zero_grad()
            self.loss.backward()
            self.learning_optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {self.loss.item():.4f}")

    def predict(self, X, steps=30):
        X = X.to(self.device)

        print("================")
        for group in self.clamped_in_optimizer.param_groups:
            for p in group['params']:
                print(p.shape, p.requires_grad, torch.sum(p).item())
        print("================")

        # Clamp input without overwriting tensor
        with torch.no_grad():
            self.activations[0].data.copy_(X)
            self.activations[-1].data.zero_()  # reset output

        self.activations[0].requires_grad_()
        self.activations[-1].requires_grad_()

        # Stabilize output and hidden layers
        self.stabilize(self.clamped_in_optimizer, steps)
        return self.activations[-1].detach()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PCNet(dims=[2, 4, 1], device=device)
    model.train(X, Y)

    print("\nFinal Weights:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} W:\n{layer.W}")
        print(f"Layer {i} b:\n{layer.b}")

    X_test = X.to(device)
    y_hat = model.predict(X_test)
    print("\nTest Inputs:\n", X_test)
    print("Predicted Outputs:\n", y_hat)

if __name__ == "__main__":
    main()
