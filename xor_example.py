# XOR data

import torch
from pcn_model import PCNet

X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PCNet(dims=[2, 4, 1], device=device)
    model.train(X, Y)

    print("\nFinal Weights:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} W:\n{layer.W}")
        print(f"Layer {i} b:\n{layer.b}")

    print("==============================")
    X_test = X.to(device)
    y_hat = model.predict(X_test)
    print("Test Inputs:\n", X_test)
    print("Predicted Outputs:\n", y_hat)

    print("==============================")
    y_test = Y.to(device)
    X_hat = model.reverse_predict(y_test)
    print("Test Outputs", y_test)
    print("Predicted Inputs: \n", X_hat)

if __name__ == "__main__":
    main()
