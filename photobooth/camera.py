import subprocess
import os
import datetime
from PIL import Image


class Camera:
    def __init__(self):
        print("Starting Camera ...")

    def snap_image(self, output_dir=None, filename=None, roi=None):
        print("Capturing image")
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = "photos/snapped"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate image file path
        if filename is None:
            image_filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            image_filename = filename + ".jpg"
            
        image_filepath = os.path.join(output_dir, image_filename)
        
        # Base libcamera command
        libcamera_command = [
            "libcamera-still", 
            "-o", image_filepath, 
            "-t", "250", 
            "-n", 
            "--sharpness=5", 
            "--autofocus-window=0.5,0.33,0.8,0.67", 
            "--autofocus-speed=normal", 
            "--lens-position=0.7",
            "--autofocus-on-capture=0"
        ]
        
        # Add Region of Interest (ROI) if provided
        if roi:
            # ROI should be a tuple or list [x0, y0, x1, y1]
            roi_cmd = "--roi=" + ",".join(map(str, roi))
            libcamera_command.append(roi_cmd)

        # Suppress log output
        with open(os.devnull, 'w') as devnull:
            try:
                subprocess.run(libcamera_command, stdout=devnull, stderr=devnull, check=True)
                print(f"🎆 Image captured and saved to {image_filepath}")
                self.crop_to_square(image_filepath)
                return image_filepath
            except subprocess.CalledProcessError as e:
                print(f"Error capturing image: {e}")
                return None
        
    def crop_to_square(self, image_filepath):
        with Image.open(image_filepath) as img:
            width, height = img.size   # Get dimensions
            new_size = min(width, height)

            left = (width - new_size)/2
            top = (height - new_size)/2
            right = (width + new_size)/2
            bottom = (height + new_size)/2
            
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(image_filepath)