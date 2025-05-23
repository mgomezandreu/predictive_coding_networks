import torch
import torch.nn as nn
import torch.optim as optim

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
        self.clamped_in_optimizer = optim.Adam(self.activations[1:], lr=0.3)
        self.clamped_out_optimizer = optim.Adam(self.activations[:-1], lr =0.3)
        self.clamped_both_optimizer = optim.Adam(self.internal_activations, lr=0.3)
        self.learning_optimizer = optim.Adam(self.parameters(), lr=0.01)

    def compute_errors_and_loss(self):
        self.errors = []
        self.loss = 0.0
        for i, layer in enumerate(self.layers):
            pred = layer(self.activations[i])
            err = self.activations[i + 1] - pred
            self.errors.append(err)
            self.loss += (err ** 2).sum()

        if getattr(self, "generating", False):
            print("Generating mode")
            # L1 sparsity penalty
            # self.loss += 0.001 * self.activations[0].abs().sum()

            # Optional: encourage values near 0 or 1 (low entropy)
            input_sigmoid = torch.sigmoid(self.activations[0])
            entropy = -(input_sigmoid * torch.log(input_sigmoid + 1e-6) + (1 - input_sigmoid) * torch.log(1 - input_sigmoid + 1e-6))
            self.loss += 0.005 * entropy.sum()

    def stabilize(self, optimizer, steps):
        self.stabilize_until_convergence(optimizer)

    def stabilize_until_convergence(self, optimizer, max_steps=100, tol=1e-3, verbose=False):
        prev_loss = float('inf')

        for step in range(max_steps):
            optimizer.zero_grad()
            self.compute_errors_and_loss()
            loss_value = self.loss.item()

            # Check for convergence
            if abs(prev_loss - loss_value) < tol:
                if verbose:
                    print(f"Converged at step {step}, loss diff: {abs(prev_loss - loss_value):.6f}")
                break

            prev_loss = loss_value
            self.loss.backward(retain_graph=True)
            optimizer.step()


    def train(self, X, Y, epochs=3000):
        X, Y = X.to(self.device), Y.to(self.device)

        for epoch in range(epochs):
            # Clamp input/output without overwriting
            with torch.no_grad():
                self.activations[0].data.copy_(X)
                self.randomize_activations(layers=[i for i in range(1, len(self.activations) - 1)])
                self.activations[-1].data.copy_(Y)

            # Inference
            self.stabilize(self.clamped_both_optimizer, steps=30)

            # Learning
            self.compute_errors_and_loss()
            self.learning_optimizer.zero_grad()
            self.loss.backward()
            self.learning_optimizer.step()


    def train_with_loader(self, data_loader, epochs=3):
        for epoch in range(epochs):
            batch_num = 0
            for X, Y in data_loader:
                if (X.shape[0] != self.batch_size):
                    continue
            
                Y = torch.nn.functional.one_hot(Y, num_classes=self.dims[-1]).float().to(self.device)

                with torch.no_grad():
                    self.activations[0].data.copy_(X)
                    self.randomize_activations(layers=[i for i in range(1, len(self.activations) - 1)])
                    self.activations[-1].data.copy_(Y)

                # Inference
                self.stabilize(self.clamped_both_optimizer, steps=30)

                # Learning
                self.compute_errors_and_loss()
                self.learning_optimizer.zero_grad()
                self.loss.backward()
                self.learning_optimizer.step()

                print(f"Batch {batch_num}, Cost: {self.loss.item():.4f}", end="\r")
                batch_num += 1

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}, Loss: {self.loss.item():.4f}")

    def randomize_activations(self, layers):
        for i in range(len(self.activations)):
            if i in layers:
                self.activations[i].data.copy_(torch.randn_like(self.activations[i]))


    def predict(self, X, steps=30):
        X = X.to(self.device)

        with torch.no_grad():
            self.activations[0].data.copy_(X)
            self.randomize_activations(layers=[i for i in range(1, len(self.activations))])

        # Stabilize output and hidden layers
        self.stabilize(self.clamped_in_optimizer, steps)
        return self.activations[-1].detach()
    
    def reverse_predict(self, y, steps = 30):
        y = y.to(self.device)

        with torch.no_grad():
            self.randomize_activations(layers=[i for i in range(0, len(self.activations) - 1)])
            self.activations[-1].data.copy_(y)
        
        self.activations[0].requires_grad_()
        self.activations[-1].requires_grad_()

        self.stabilize(self.clamped_out_optimizer, steps)
        return self.activations[0].detach()



