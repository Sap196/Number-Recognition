from main import *
from tekenvlak import teken_cijfer

model, loss_function, optimizer = init_model()
batch = load_existing_modell(model=model)
image = teken_cijfer()

if image is not None:
    print("Voorspelling:", predict(model, image))