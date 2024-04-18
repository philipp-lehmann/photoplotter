import cv2
import numpy as np
import os
import svgwrite
from lxml import etree

class ImageParser:
    def __init__(self):
        self.maxTrials = 10
        print("Starting ImageParser ...")
        pass
    
    def detect_faces(self, image_filepath):
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
        
    def enhance_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        mask = np.zeros_like(image)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)  # width and height of the ellipse axes
            mask = cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        # Soften the edges of the mask
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Create a 3-channel mask for color image blending
        mask_channels = cv2.split(blurred_mask)
        mask_for_blending = cv2.merge(mask_channels)

        # Blend the original image with the mask using a weighted sum
        # Note: ensure the masks and image are of the same data type (e.g., uint8)
        enhanced_image = cv2.addWeighted(image, 1, mask_for_blending, 0.5, 0)

        return enhanced_image

    def apply_local_enhancements(self, roi):
        # Convert to YUV color space
        img_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        # Apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        # Convert back to BGR color space
        enhanced_roi = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return enhanced_roi

    def optimize_image(self, img):
        # Convert to YUV color space
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Apply histogram equalization on the luminance channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # Convert back to BGR color space
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Optionally, apply additional enhancements like HDR effect here

        # Return the optimized image
        return img_output

    def convert_to_svg(self, image_filepath, target_width=640, target_height=640, scale_x=0.35, scale_y=0.35, min_paths=30, max_paths=90, min_contour_area=20, suffix=''):
        print("Converting current photo to SVG")
        if os.path.isfile(image_filepath):
            # Load the image using OpenCV
            image = cv2.imread(image_filepath)
            if image is not None:
                # Resize the image to the target dimensions
                image = cv2.resize(image, (target_width, target_height))
                
                opt_image = self.optimize_image(image)
                optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized.' + image_filepath.rsplit('.', 1)[1]
                cv2.imwrite(optimized_image_path, opt_image)
                # print(f"Optimized image saved to: {optimized_image_path}")
                
                # opt_image = self.enhance_faces(opt_image)
                # optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized_faces.' + image_filepath.rsplit('.', 1)[1]
                # cv2.imwrite(optimized_image_path, opt_image)
                # print(f"Optimized image saved to: {optimized_image_path}")
                
                # Convert the image to grayscale for edge detection
                gray_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2GRAY)
                # Perform initial edge detection
                edges = cv2.Canny(gray_image, threshold1=80, threshold2=200)
                # Find initial contours based on edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Filter out small contours
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
                # Initially simplify contours to reduce complexity
                simplified_contours = [cv2.approxPolyDP(contour, 1, True) for contour in filtered_contours]

                # Initialize variables for the adjustment loop
                num_paths = 0
                trials = 0
                lower_threshold = 80
                upper_threshold = 200
                epsilon = 1

                # Adjustment loop
                while num_paths < min_paths or num_paths > max_paths:
                    dwg = svgwrite.Drawing(size=(target_width, target_height))
                    num_paths = self.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)

                    if num_paths < min_paths or num_paths > max_paths:
                        print(f"{min_contour_area} - {trials}: {num_paths}")
                        lower_threshold, upper_threshold, epsilon = self.adjust_thresholds(
                            num_paths, min_paths, max_paths, lower_threshold, upper_threshold, epsilon, trials)
                        edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        simplified_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

                    trials += 1
                    if trials > self.maxTrials:
                        break
                    
                print(f"{min_contour_area} - {trials}: {num_paths}")
                
                # Sort contours by their distance from the image center
                image_center = np.array([target_width // 2, target_height // 2])
                sorted_contours = sorted(simplified_contours, key=lambda c: np.linalg.norm(np.mean(np.squeeze(c), axis=0) - image_center))

                # Limit the number of paths if necessary
                if len(sorted_contours) > max_paths:
                    simplified_contours = sorted_contours[:max_paths]
                else:
                    simplified_contours = sorted_contours

                num_paths = len(simplified_contours)

                # Print the number of paths after processing and before saving
                print(f"Creating an image with {num_paths} paths.")
                dwg = svgwrite.Drawing(size=(target_width, target_height))
                num_paths = self.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)

                # Save to output folder
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(parent_dir, "photos/traced")
                os.makedirs(output_dir, exist_ok=True)
                
                svg_filename = os.path.splitext(os.path.basename(image_filepath))[0] + suffix + '.svg'
                svg_filepath = os.path.join(output_dir, svg_filename)
                dwg.saveas(svg_filepath)
                
                # Return file path
                return svg_filepath
        # Return None if the file doesn't exist or no image is loaded
        return None

    def create_output_svg(self, image_svg_path, scale_factor = 0.8, offset_x=0, offset_y=0, id=0):
        # Load the original SVG content from a file
        with open(image_svg_path, 'rb') as file:  # Note 'rb' mode for reading as bytes
            svg_data = file.read()

        # Parse the original SVG
        root = etree.fromstring(svg_data)

        # Create a new SVG drawing with svgwrite, setting the artboard size to 420x297mm (check size)
        dwg = svgwrite.Drawing(size=('1587', '1122'))

        # Transform and position the image
        group = dwg.g(id="all_paths", transform=f"translate({offset_x}, {offset_y}) scale({scale_factor})")
        # dwg.add(dwg.rect(insert=(0, 0), size=('1587', '1122px'), fill='white'))

        for element in root.iter("{http://www.w3.org/2000/svg}*"):
            if element.tag.endswith('polyline'):
                points = element.get('points')
                if points:  # Ensure there's a points attribute
                    # Prepare the points in the format expected by svgwrite
                    points_list = points.strip().split(" ")
                    # Convert each point from 'x,y' to (x, y) tuple format
                    points_tuples = [tuple(map(float, p.split(','))) for p in points_list if ',' in p]
                    if points_tuples:  # Ensure there's at least one valid point
                        group.add(dwg.polyline(points=points_tuples, stroke='black', fill='none'))

        dwg.add(group)
        
         # Use an absolute path for the output directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "photos/current")  # Join it with your relative path
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # svg_filename = os.path.splitext(os.path.basename(image_svg_path))[0] + str(id) + '.svg'
        svg_filename = 'temp-' + str(id) + '.svg'
        output_svg_path = os.path.join(output_dir, svg_filename)  # This is your absolute path for the SVG file
        dwg.saveas(output_svg_path)

        return output_svg_path

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
    def adjust_thresholds(num_paths, min_paths, max_paths, lower_threshold, upper_threshold, epsilon, trial):
        # Adjusts the thresholds based on the number of paths
        
        if num_paths < min_paths:
            lower_threshold -= 5
            upper_threshold -= 10 
            epsilon *= 0.85 
        elif num_paths > max_paths:
            lower_threshold += 5 
            upper_threshold += 15 
            epsilon *= 1.15
            
        return lower_threshold, upper_threshold, epsilon