import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
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
        self.W = nn.Parameter(torch.ones(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.W.t()) + self.b)

# Predictive coding network
class PCNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.layer1 = PredictiveCodingLayer(2, 4)
        self.layer2 = PredictiveCodingLayer(4, 1)
        self.to(device)  # move model parameters to device

    def inference(self, x, steps=30, lr=0.1):
        x = x.to(self.device)
        x1 = torch.randn(x.shape[0], 4, requires_grad=True, device=self.device)
        x2 = torch.randn(x.shape[0], 1, requires_grad=True, device=self.device)

        for _ in range(steps):
            mu1 = self.layer1(x)
            mu2 = self.layer2(x1)

            e1 = x1 - mu1
            e2 = x2 - mu2

            loss = ((e1**2).sum(dim=1) + (e2**2).sum(dim=1)).sum()
            grads = torch.autograd.grad(loss, [x1, x2], retain_graph=True)

            x1 = (x1 - lr * grads[0]).detach().requires_grad_()
            x2 = (x2 - lr * grads[1]).detach().requires_grad_()

        return x1.detach(), x2.detach()

    def inference_with_adam(self, x, steps=30, lr=0.05):
        x = x.to(self.device)
        x1 = torch.randn(x.size(0), 4, requires_grad=True, device=self.device)
        x2 = torch.randn(x.size(0), 1, requires_grad=True, device=self.device)

        optimizer = torch.optim.Adam([x1, x2], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            mu1 = self.layer1(x)
            mu2 = self.layer2(x1)

            e1 = x1 - mu1
            e2 = x2 - mu2

            loss = ((e1**2).sum(dim=1) + (e2**2).sum(dim=1)).sum()
            loss.backward()
            optimizer.step()

        return x1.detach(), x2.detach()

    def inference_with_clamped_out(self, x, y, steps=80, lr=0.2):
        x = x.to(self.device)
        y = y.to(self.device)

        x1 = torch.randn(x.size(0), 4, requires_grad=True, device=self.device)
        x2 = y.detach()

        optimizer = torch.optim.Adam([x1], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            mu1 = self.layer1(x)
            mu2 = self.layer2(x1)

            e1 = x1 - mu1
            e2 = x2 - mu2

            loss = ((e1**2).sum(dim=1) + (e2**2).sum(dim=1)).sum()
            loss.backward()
            optimizer.step()

        # Final error after convergence
        final_mu1 = self.layer1(x)
        final_mu2 = self.layer2(x1.detach())
        final_e1 = x1.detach() - final_mu1
        final_e2 = x2 - final_mu2

        total_loss = (final_e1**2).mean() + (final_e2**2).mean()
        return total_loss

    def train(self, X, Y):
        X, Y = X.to(self.device), Y.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        for epoch in range(3000):
            optimizer.zero_grad()
            loss = self.inference_with_clamped_out(X, Y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = PCNet(device)

    model.train(X, Y)

    print("\nFinal weights W1:", model.layer1.W)
    print("Final biases b1:", model.layer1.b)
    print("Final weights W2:", model.layer2.W)
    print("Final biases b2:", model.layer2.b)

    X_test = X.to(device)
    _, y_hat = model.inference(X_test)
    print("\nInputs:\n", X_test)
    print("y_hat:\n", y_hat.round())

if __name__ == "__main__":
    main()
