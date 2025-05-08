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

# Define predictive coding layer
class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.ones(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.W.t()) + self.b)
    

    

# Simple predictive coding model with one hidden layer
class PCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PredictiveCodingLayer(2, 4)
        self.layer2 = PredictiveCodingLayer(4, 1)

    def inference(self, x, steps=30, lr=0.1):
        x1 = torch.randn(x.shape[0], 4, requires_grad=True)
        x2 = torch.randn(x.shape[0], 1, requires_grad=True)

        for _ in range(steps):
            mu1 = self.layer1(x)     # shape: (1, 4)
            mu2 = self.layer2(x1)    # shape: (1, 1)

            e1 = x1 - mu1
            e2 = x2 - mu2

            loss = ((e1**2).sum(dim=1) + (e2**2).sum(dim=1)).sum()

            grads = torch.autograd.grad(loss, [x1, x2], retain_graph=True)
            x1 = (x1 - lr * grads[0]).detach().requires_grad_()
            x2 = (x2 - lr * grads[1]).detach().requires_grad_()

        return x1.detach(), x2.detach()
    
    def inference_with_clamped_out(self, x, y,steps=30, lr=0.1):
        x1 = torch.randn(x.shape[0], 4, requires_grad=True)
        x2 = y
        for _ in range(steps):
            mu1 = self.layer1(x)     # shape: (1, 4)
            mu2 = self.layer2(x1)    # shape: (1, 1)

            e1 = x1 - mu1
            e2 = x2 - mu2

            loss = ((e1**2).sum(dim=1) + (e2**2).sum(dim=1)).sum()

            grads = torch.autograd.grad(loss, [x1], retain_graph=True)
            x1 = (x1 - lr * grads[0]).detach().requires_grad_()

        total_loss = (e1**2).mean() + (e2**2).mean()

        return total_loss 
    
    def train(self, X,Y):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(3000):
            optimizer.zero_grad()
            loss = self.inference_with_clamped_out(X, Y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                # Print loss every 100 epochs
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def main():
    # Initialize model
    model = PCNet()
    # Train the model
    model.train(X, Y)

    print("Final weights W:", model.layer1.W)
    print("Final biases b:", model.layer1.b)
    print("Final weights W2:", model.layer2.W)  
    print("Final biases b2:", model.layer2.b)

    print(X)
    _, y_hat = model.inference(X)
    print("y_hat:", y_hat)
    
    
if __name__ == "__main__":
    main()
        