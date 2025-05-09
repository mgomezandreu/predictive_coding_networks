import os
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
from pcn_model import PCNet

from PIL import Image

from scipy.ndimage import gaussian_filter


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

    if os.path.exists("pcn_mnist.pth"):
        model.load_state_dict(torch.load("pcn_mnist.pth"))
        print("Model loaded from file.")

    else:
        print("No pre-trained model found. Training a new one.")
        model.train_with_loader(train_loader, epochs=3)

        torch.save(model.state_dict(), "pcn_mnist.pth")
        print("Model saved to file.")

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

    # do some reverse prediction, by creating a random output
    Y = torch.randint(0, 10, (model.batch_size,))
    Y_encoded = torch.nn.functional.one_hot(Y, num_classes=model.dims[-1]).float().to(device)

    model.generating = True
    X = model.reverse_predict(Y_encoded, steps=30)
    X = X.view(-1, 1, 28, 28).to("cpu")
    X = X.detach().numpy()

    # Save the generated images

    # deletate the old images
    if os.path.exists("imgs"):
        for file in os.listdir("imgs"):
            os.remove(os.path.join("imgs", file))
    else:
        os.makedirs("imgs")

    for i in range(X.shape[0]):
        img = X[i].squeeze()
        img = 1 / (1 + np.exp(-img)) # Apply sigmoid to get values in [0, 1]
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        # apply blur filter\

        img = gaussian_filter(img, sigma=1)

        # #invert the image
        # img = 255 - img

        
        img_path = f"imgs/generated_image_{i}_Label_{Y[i]}.png"


        Image.fromarray(img).save(img_path)
        print(f"Generated image saved to {img_path}")

  

if __name__ == "__main__":
    main()
