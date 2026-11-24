from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def make_loaders():
    """Download MNIST indien nodig en maak loaders voor training en toetsing."""
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=32, shuffle=True),
        DataLoader(test_data, batch_size=32, shuffle=False),
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    def choose_output(self, state):
        return torch.argmax(self(state), dim=1)


def load_existing_modell(model) -> int:
    checkpoints = []
    for path in MODEL_DIR.glob("number_recognition_*.pt"):
        batch_text = path.stem.removeprefix("number_recognition_")
        if batch_text.isdecimal():
            checkpoints.append((int(batch_text), path))

    if not checkpoints:
        return 0

    batch, latest_path = max(checkpoints, key=lambda checkpoint: checkpoint[0])
    weights = torch.load(latest_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    print(f"Model geladen uit {latest_path} (batch {batch})")
    return batch


def init_model():
    model = NeuralNetwork()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_function, optimizer


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model.choose_output(images)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    print(f"Testnauwkeurigheid: {correct / total:.1%}")


def train(batch, model, optimizer, loss_function, train_loader, save_checkpoints=True):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()

        batch += 1
        print(f"Batch {batch} | Loss: {loss.item():.4f}")
        if save_checkpoints and batch % 500 == 0:
            torch.save(model.state_dict(), MODEL_DIR / f"number_recognition_{batch}.pt")
    return batch


def run_model(train_model=True, evaluate_model=True, load_existing_model=True, save_checkpoints=True):
    train_loader, test_loader = make_loaders()
    model, loss_function, optimizer = init_model()
    batch = load_existing_modell(model) if load_existing_model else 0

    if save_checkpoints:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if train_model:
        batch = train(batch, model, optimizer, loss_function, train_loader, save_checkpoints)
    if save_checkpoints and train_model:
        torch.save(model.state_dict(), MODEL_DIR / f"number_recognition_{batch}.pt")
    if evaluate_model:
        evaluate(model, test_loader)


def predict(model, image):
    model.eval()
    with torch.inference_mode():
        return model.choose_output(image)


if __name__ == "__main__":
    run_model()
