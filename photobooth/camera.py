from .imageparser import ImageParser
import subprocess
import os
import time
import datetime
from PIL import Image


class Camera:
    def __init__(self):
        print("Starting Camera ...")
        # Assuming ImageParser is correctly implemented elsewhere
        self.image_parser = ImageParser()

    def snap_image(self, output_dir=None, filename=None, roi=[0.33, 0.33, 0.67, 0.67]):
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

        libcamera_command = ["libcamera-still", "-o", image_filepath, "-t", "750", "-n"]


        # Set the Region of interest for autofocus if provided
        if roi:
            # ROI should be a tuple or list [x0, y0, x1, y1]
            roi_cmd = "--roi=" + ",".join(map(str, roi))
            libcamera_command.append(roi_cmd)

        try:
            subprocess.run(libcamera_command, check=True)
            print(f"Image saved to {image_filepath}")
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
            print(f"Image cropped to square and saved to {image_filepath}")

    def process_image(self, image_filepath):
        # Implement processing using ImageParser
        pass
