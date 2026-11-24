from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path

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
batch = 0

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

def get_model():
    model = NeuralNetwork()
    return model

def load_existing_modell(model) -> int:
    checkpoints = []
    for path in Path("models").glob("number_recognition_*.pt"):
        batch_text = path.stem.removeprefix("number_recognition_")
        if batch_text.isdecimal():
            checkpoints.append((int(batch_text), path))

    if checkpoints:
        batch, latest_path = max(checkpoints, key=lambda checkpoint: checkpoint[0])
        weights = torch.load(latest_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        print(f"Model geladen uit {latest_path} (batch {batch})")

    return batch

def init_loss_and_optimize(model):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return loss_function, optimizer

def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model.choose_output(images)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    print(f"Testnauwkeurigheid: {correct / total:.1%}")


def train(batch, model, optimizer, loss_function, save_best_model=True):
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()

        batch += 1
        print(f"Batch {batch} | Loss: {loss.item():.4f}")

        if save_best_model:
            if batch % 500 == 0:
                torch.save(model.state_dict(), f"models/number_recognition_{batch}.pt")

    return batch

def run_model(train_model=True, evaluate_model=True, load_existing_model=True, save_best_model=True):
    model, loss_function, optimizer = init_model()

    if load_existing_model:
        batch = load_existing_modell(model=model)
    else:
        batch = 0

    if train_model:    
        batch = train(batch=batch, model=model, optimizer=optimizer, loss_function=loss_function, save_best_model=save_best_model)

    if save_best_model:
        torch.save(model.state_dict(), f"models/number_recognition_{batch}.pt")

    if evaluate_model:
        evaluate(model=model)

def init_model():
    model = get_model()
    loss_function, optimizer = init_loss_and_optimize(model)

    return model, loss_function, optimizer

def predict(model, image):
    model.eval()
    with torch.inference_mode():
        prediction = model.choose_output(image)

    return prediction