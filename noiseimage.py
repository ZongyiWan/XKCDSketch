import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
image_path = "image7.png"  # Change this to your image path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Ensure output directory exists
output_dir = "noise_series"
os.makedirs(output_dir, exist_ok=True)

# Parameters
num_steps = 10  # Number of noise steps
max_noise_intensity = 255  # Maximum noise intensity for full static

# Generate and save images with increasing noise
for i in range(num_steps + 1):
    noise_ratio = i / num_steps  # Linearly scale noise from 0 to full static
    noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)  # Pure noise
    noisy_image = cv2.addWeighted(image, 1 - noise_ratio, noise, noise_ratio, 0)  # Blend noise

    # Save the noisy image
    output_path = os.path.join(output_dir, f"noisy_{i:02d}.png")
    pure_noise = cv2.subtract(noisy_image, image)
    # cv2.imwrite(output_path, cv2.cvtColor(noise, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

# Display a sample of images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for j, idx in enumerate(np.linspace(0, num_steps, 5, dtype=int)):
    img = cv2.imread(os.path.join(output_dir, f"noisy_{idx:02d}.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[j].imshow(img)
    axes[j].axis("off")
    axes[j].set_title(f"Step {idx}")
plt.show()