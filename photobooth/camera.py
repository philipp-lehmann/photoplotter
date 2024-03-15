from .imageparser import ImageParser

class Camera:
    def __init__(self):
        self.image_parser = ImageParser()

    def capture_image(self):
        # Capture an image
        print(f"Capturing image")

    def release(self):
        # Release the camera resource
        self.cap.release()
