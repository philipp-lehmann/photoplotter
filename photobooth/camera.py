from .imageparser import ImageParser
import subprocess
import os
import datetime

class Camera:
    def __init__(self):
        print("Starting Camera ...")
        # Assuming ImageParser is correctly implemented elsewhere
        self.image_parser = ImageParser()

    def capture_image(self):
        print("Capturing image")
        image_dir = "photos/captured"
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_filepath = os.path.join(image_dir, image_filename)

        # Use libcamera-still to capture an image
        try:
            subprocess.run(["libcamera-still", "-o", image_filepath, "-t", "100"], check=True)  # 100ms timeout

            print(f"Image saved to {image_filepath}")
            return image_filepath
        except subprocess.CalledProcessError as e:
            print(f"Error capturing image: {e}")
            return None

    def process_image(self, image_filepath):
        # Implement processing using ImageParser
        pass
