from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
from pcn_model import PCNet

def get_mnist_data_loaders():
    """
    Returns the MNIST dataset loaders for training and testing.
    """
    # Transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape: [1,28,28]
        transforms.Lambda(lambda x: x.view(-1))  # flatten to [784]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def main():
    train_loader, test_loader = get_mnist_data_loaders()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    model = PCNet(dims=[784, 100, 10], device=device, batch_size=64)

    model.train_with_loader(train_loader, epochs=3)

    # evaluate accuracy
    correct = 0
    total = 0

   
    for X, Y in test_loader:
        if (X.shape[0] != model.batch_size):
            continue
        # Y_encoded = torch.nn.functional.one_hot(Y, num_classes=model.dims[-1]).float().to(device)
        # print(f"Y shape after one-hot encoding: {Y.shape}")
        y_hat = model.predict(X).to("cpu")
        y_hat = torch.argmax(y_hat, dim=1)
        correct += (y_hat == Y).sum().item()
        total += Y.shape[0]
        accuracy = correct / total

        # print(f"Batch {total // model.batch_size}: Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
