"""Teken cijfers en bekijk hun voorspelling in hetzelfde venster."""

import tkinter as tk

from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor


def _to_mnist_tensor(drawing):
    bounds = drawing.getbbox()
    if bounds is None:
        raise ValueError("Teken eerst een cijfer.")

    digit = drawing.crop(bounds)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    image = Image.new("L", (28, 28), 0)
    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    image.paste(digit, (left, top))
    return ToTensor()(image)


def teken_cijfer(voorspel):
    """Roep voorspel(tensor) aan voor elke klik op Voorspel."""
    size = 280
    radius = 9
    drawing = Image.new("L", (size, size), 0)
    brush = ImageDraw.Draw(drawing)
    previous = None

    window = tk.Tk()
    window.title("Teken een cijfer")
    window.resizable(False, False)

    canvas = tk.Canvas(
        window, width=size, height=size, bg="black", highlightthickness=0
    )
    canvas.pack(padx=12, pady=(12, 8))

    status = tk.Label(window, text="Teken één cijfer met je muis.")
    status.pack()

    def paint(event):
        nonlocal previous
        x, y = point = (event.x, event.y)
        if previous is not None:
            canvas.create_line(*previous, x, y, fill="white", width=2 * radius)
            brush.line((previous, point), fill=255, width=2 * radius)

        dot = (x - radius, y - radius, x + radius, y + radius)
        canvas.create_oval(*dot, fill="white", outline="")
        brush.ellipse(dot, fill=255)
        previous = point

    def start_stroke(event):
        nonlocal previous
        previous = None
        paint(event)

    def clear():
        nonlocal previous
        previous = None
        canvas.delete("all")
        drawing.paste(0, (0, 0, size, size))
        status.config(text="Teken één cijfer met je muis.")

    def submit():
        if drawing.getbbox() is None:
            status.config(text="Teken eerst een cijfer.")
            return
        prediction = voorspel(_to_mnist_tensor(drawing))
        status.config(text=f"Voorspelling: {prediction}")

    canvas.bind("<Button-1>", start_stroke)
    canvas.bind("<B1-Motion>", paint)

    buttons = tk.Frame(window)
    buttons.pack(pady=(8, 12))
    tk.Button(buttons, text="Wis", command=clear).pack(side="left", padx=4)
    tk.Button(buttons, text="Voorspel", command=submit).pack(
        side="left", padx=4
    )

    window.mainloop()
