import os
from PIL import Image
import colorsys

# Input and output folders
input_folder = "display"
output_folder = "display_corrected"

# Hue shift in degrees (change this value)
HUE_SHIFT_DEGREES = 234
HUE_SHIFT = HUE_SHIFT_DEGREES / 360.0  # convert to [0,1] range

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def shift_hue(img, shift):
    img = img.convert("RGB")
    pixels = img.load()

    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pixels[x, y]
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)

            # Shift hue
            h = (h + shift) % 1.0

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))

    return img

# Process all JPGs in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = Image.open(input_path)
        shifted = shift_hue(img, HUE_SHIFT)
        
        # Save with maximum quality
        shifted.save(output_path, quality=100, subsampling=0, optimize=True)

print(f"Hue shift of {HUE_SHIFT_DEGREES}° applied. Files saved in: {output_folder}")
