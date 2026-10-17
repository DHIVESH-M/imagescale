import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = Image.open("input.png").convert("RGB")
pixels = np.array(image)
print("Image shape:", pixels.shape)
print("Data type:", pixels.dtype)
gray = (
    0.299 * pixels[:, :, 0]
    + 0.587 * pixels[:, :, 1]
    + 0.114 * pixels[:, :, 2]
).astype(np.uint8)
brightness = np.clip(
    pixels.astype(np.int16) + 40, 0, 255
).astype(np.uint8)
flipped = pixels[:, ::-1, :]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(pixels)
axes[0, 0].set_title("Original")
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale")
axes[1, 0].imshow(brightness)
axes[1, 0].set_title("Brighter")
axes[1, 1].imshow(flipped)
axes[1, 1].set_title("Flipped")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()
