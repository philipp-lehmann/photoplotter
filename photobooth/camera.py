from .imageparser import ImageParser

class Camera:
    def __init__(self):
        print(f"Starting Camera ...")
        self.image_parser = ImageParser()

    def capture_image(self):
        # Capture an image
        print(f"Capturing image")

    def process_image(self, image_filepath):
        # Process the image and convert it to SVG format
        svg_filepath, num_paths = self.image_parser.convert_to_svg(image_filepath)
        if svg_filepath:
            print(f"Image processed successfully. SVG saved at: {svg_filepath}")
        else:
            print("Failed to process the image.")

        return svg_filepath, num_paths

    def release(self):
        # Release the camera resource
        self.cap.release()
