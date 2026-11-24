from main import *
from tekenvlak import teken_cijfer

model, _, _ = init_model()
batch = load_existing_modell(model)

if batch == 0:
    raise FileNotFoundError("Geen getraind checkpoint gevonden in de map models.")

teken_cijfer(lambda image: predict(model, image.unsqueeze(0)).item())