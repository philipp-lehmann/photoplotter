import cv2
import os
import svgwrite

class ImageParser:
    def __init__(self):
        print("Starting ImageParser ...")
        pass
    
    def detect_faces(self, image_filepath):
        """
        Detect faces in an image.

        Args:
            image_filepath (str): The path to the image file.

        Returns:
            bool: True if at least one face is detected, False otherwise.
        """
        # Load the pre-trained Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load the image
        image = cv2.imread(image_filepath)
        if image is None:
            print("Failed to load image.")
            return False
        
        # Convert the image to grayscale (required for face detection)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Check if any faces are detected
        if len(faces) > 0:
            print(f"Detected {len(faces)} face(s) in the image.")
            return True
        else:
            print("No faces detected in the image.")
            return False

    def convert_to_svg(self, image_filepath, target_width=640, target_height=640, scale_x=0.35, scale_y=0.35, min_paths=30, max_paths=75):
        print("Converting current photo to SVG")
        if os.path.isfile(image_filepath):
            # Load the image using OpenCV
            image = cv2.imread(image_filepath)
            if image is not None:
                # Resize the image to the target dimensions
                image = cv2.resize(image, (target_width, target_height))
                # Convert the image to grayscale for edge detection
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Perform initial edge detection
                edges = cv2.Canny(gray_image, threshold1=80, threshold2=200)
                # Find initial contours based on edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Initially simplify contours to reduce complexity
                simplified_contours = [cv2.approxPolyDP(contour, 1, True) for contour in contours]

                # Initialize variables for the adjustment loop
                num_paths = 0
                trials = 0
                lower_threshold = 80
                upper_threshold = 200
                epsilon = 1

                # Loop to adjust thresholds until the number of paths is within desired range
                while num_paths < min_paths or num_paths > max_paths:
                    # Create a new SVG drawing for each iteration to avoid duplicating paths
                    dwg = svgwrite.Drawing(size=(target_width, target_height))

                    # Add current contours to the SVG drawing
                    num_paths = ImageParser.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)

                    # If the number of paths is not within the desired range, adjust the thresholds
                    if num_paths < min_paths or num_paths > max_paths:
                        lower_threshold, upper_threshold, epsilon = ImageParser.adjust_thresholds(
                            num_paths, min_paths, max_paths, lower_threshold, upper_threshold, epsilon, gray_image)

                        # Perform edge detection with adjusted thresholds
                        edges = cv2.Canny(gray_image, threshold1=lower_threshold, threshold2=upper_threshold)
                        # Find new contours with the adjusted thresholds
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Simplify contours again with the adjusted epsilon
                        simplified_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

                    # Limit the number of adjustment trials to prevent infinite loops
                    trials += 1
                    if trials > 10:
                        break

                # After finalizing the contours, print the outcome
                print(f"Settled on an image with {num_paths} paths.")

                # Save the SVG file to the traced directory
                output_dir = "photos/traced"
                os.makedirs(output_dir, exist_ok=True)
                svg_filename = os.path.splitext(os.path.basename(image_filepath))[0] + '.svg'
                svg_filepath = os.path.join(output_dir, svg_filename)
                dwg.saveas(svg_filepath)

                return svg_filepath
        # Return None if the file doesn't exist or no image is loaded
        return None

    @staticmethod
    def add_contours_to_svg(dwg, contours, scale_x, scale_y):
        # Implementation for adding contours to the SVG with styling
        num_paths = 0
        for contour in contours:
            # Convert contour points to a format suitable for svgwrite and apply scaling
            points = [(point[0][0] * scale_x, point[0][1] * scale_y) for point in contour]
            dwg.add(dwg.polyline(points, fill="none", stroke="#aaa", stroke_width="1"))
            num_paths += 1
        return num_paths

    @staticmethod
    def adjust_thresholds(num_paths, min_paths, max_paths, lower_threshold, upper_threshold, epsilon, gray_image):
        # Adjusts the thresholds based on the number of paths
        if num_paths < min_paths:
            lower_threshold -= 10
            upper_threshold -= 10
            epsilon *= 0.9
        elif num_paths > max_paths:
            lower_threshold += 5
            upper_threshold += 15
            epsilon *= 1.1
        return lower_threshold, upper_threshold, epsilon
