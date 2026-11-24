from main import init_model, load_existing_modell, predict
from tekenvlak import teken_cijfer


def main():
    model, _, _ = init_model()
    batch = load_existing_modell(model)
    if batch == 0:
        raise FileNotFoundError("Geen getraind checkpoint gevonden in de map models. Voer eerst main.py uit.")
    teken_cijfer(lambda image: predict(model, image.unsqueeze(0)).item())


if __name__ == "__main__":
    main()
