from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

# Zet afbeeldingen om naar tensors met pixelwaarden tussen 0 en 1.
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x= torch.flatten(x, start_dim=1)
        x = torch.relu(self.layer1(x))

        return self.layer2(x)

    def choose_output(self, state):
        return torch.argmax(self(state), dim=1)


model = NeuralNetwork()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for images, labels in train_loader:
    # Verwijder gradiënten van de vorige batch.
    optimizer.zero_grad()

    # Bereken de tien scores per afbeelding en de fout.
    scores = model(images)
    loss = loss_function(scores, labels)

    # Bereken hoe de gewichten moeten veranderen.
    loss.backward()

    # Pas de gewichten aan.
    optimizer.step()

    print("Loss:", loss.item())

    break