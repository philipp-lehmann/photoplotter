import cv2
import os
import dlib
import numpy as np
import svgwrite
import re
import random
from lxml import etree

class ImageParser:
    def __init__(self):
        self.maxTrials = 10
        # Initialize dlib face detector and shape predictor (68 landmarks)
        self.face_detector = dlib.get_frontal_face_detector()
        base_path = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(base_path, 'shape_predictor', 'shape_predictor_68_face_landmarks.dat')
        self.landmark_detector = dlib.shape_predictor(predictor_path)
        print("Starting ImageParser ...")
        pass    
    
    def detect_faces(self, image_filepath):
        """Quick method to check if a face is present in the image"""
        image = cv2.imread(image_filepath)
        if image is None:
            print("Failed to load image.")
            return False
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image)
        return len(faces) > 0
    
    def process_face_image(self, image, target_width=800, target_height=800):
        """Optimized method that detects, crops, enhances the face and draws facial features"""
        if image is None:
            print("Failed to load image.")
            return None
        
        # Detect faces
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image)

        if len(faces) == 0:
            print("No faces detected.")

            # Crop the image by 15% on each side
            height, width = image.shape[:2]
            crop_x = int(0.2 * width)  # 15% of the width
            crop_y = int(0.2 * height)  # 15% of the height

            # Perform the cropping
            cropped_image = image[crop_y:height - crop_y, crop_x:width - crop_x]
            enhanced_image = self.enhance_faces(cropped_image)
            return enhanced_image

        # Sort faces by size (width * height) in descending order
        faces = sorted(faces, key=lambda rect: rect.width() * rect.height(), reverse=True)

        # Process up to 3 largest faces
        num_faces_to_process = min(3, len(faces))  # Process up to 3 faces

        print(f"Faces found {num_faces_to_process}")
        
        for i in range(num_faces_to_process):
            largest_face = faces[i]

            # Detect landmarks and draw them on the original image (not just the cropped one)
            self.draw_facial_landmarks(image, largest_face)
            
            # Crop the region around the face
            cropped_image = self.crop_to_largest_face(image, largest_face, target_width, target_height)
            
        # Return the original image with landmarks drawn on it
        enhanced_image = self.enhance_faces(cropped_image)
        return enhanced_image

    def crop_to_largest_face(self, image, face_rect, target_width=800, target_height=800):
        """Crop the image around the detected face to a square size."""
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        center_x, center_y = x + w // 2, y + h // 2

        # Determine the size of the square crop
        crop_size = max(w, h)
        margin = int(crop_size * 0.60)  # Add some margin around the face
        crop_size += 2 * margin

        # Calculate crop boundaries
        x_start = max(center_x - crop_size // 2, 0)
        y_start = max(center_y - crop_size // 2, 0)
        x_end = min(x_start + crop_size, image.shape[1])
        y_end = min(y_start + crop_size, image.shape[0])

        # Adjust start positions if end positions exceed image boundaries
        x_start = max(x_end - crop_size, 0)
        y_start = max(y_end - crop_size, 0)

        # Extract the cropped face region
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Pad the image if it's not square (this happens when crop touches image boundaries)
        if cropped_image.shape[0] != cropped_image.shape[1]:
            target_shape = (crop_size, crop_size, 3)
            padded_image = np.zeros(target_shape, dtype=np.uint8)
            padded_image[:cropped_image.shape[0], :cropped_image.shape[1], :] = cropped_image
            cropped_image = padded_image

        return cv2.resize(cropped_image, (target_width, target_height))

    def enhance_faces(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_l_channel = clahe.apply(l_channel)
        enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
        blended_image = cv2.addWeighted(image, 0.65, enhanced_image, 0.35, 0)
        return blended_image

    def draw_facial_landmarks(self, image, face_rect):
        """Draw the 68 facial landmarks using dlib's shape predictor."""
        if image is None or image.size == 0:
            print("Error: Image is empty or not loaded properly.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmark_detector(gray, face_rect)

        # Define probabilities for each scenario
        probability_eyebrows = 0.14 
        probability_mouth = 0.12 
        probability_jawline = 0.07
        probability_eyes_1 = 0.21
        probability_eyes_2 = 0.15
        probability_nose_1 = 0.24
        probability_nose_2 = 0.11
        probability_teeth = 0.07

        # Randomly decide whether to draw each feature based on probabilities
        if random.random() < probability_jawline: #leftjaw
            self.draw_feature_line(image, landmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8])

        if random.random() < probability_eyebrows: #eybrows
            self.draw_feature_line(image, landmarks, [17, 18, 19, 20, 21])
            self.draw_feature_line(image, landmarks, [22, 23, 24, 25, 26])

        if random.random() < probability_nose_1: #roundnose
            self.draw_feature_line(image, landmarks, [32, 33, 34, 35])
            
        if random.random() < probability_nose_2: #verticalnose
            self.draw_feature_line(image, landmarks, [27, 28, 29, 30, 33])

        if random.random() < probability_eyes_1: #eyes
            self.draw_feature_line(image, landmarks, [36, 37, 38, 39])
            self.draw_feature_line(image, landmarks, [40, 41])
            self.draw_feature_line(image, landmarks, [42, 43, 44, 45])
            self.draw_feature_line(image, landmarks, [46, 47])
        
        if random.random() < probability_eyes_2: #cross_eyes
            self.draw_feature_line(image, landmarks, [37, 40])
            self.draw_feature_line(image, landmarks, [41, 38])
            self.draw_feature_line(image, landmarks, [43, 46])                     
            self.draw_feature_line(image, landmarks, [47, 44])                     

        if random.random() < probability_mouth: #mouth
            self.draw_feature_line(image, landmarks, [60, 61, 62, 63, 64, 65, 66, 67, 60]) 
        
        if random.random() < probability_teeth: #teeth
            self.draw_feature_line(image, landmarks, [61, 67, 62, 66, 63, 65])

    def draw_feature_line(self, img, landmarks, points_indices, color=(255, 255, 255), thickness=1):
        """Helper method to draw lines connecting facial landmarks."""
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in points_indices if 0 <= i < 68]

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, thickness)

    def convert_to_svg(self, image_filepath, target_width=800, target_height=800, scale_x=0.35, scale_y=0.35, min_paths=30, max_paths=90, min_contour_area=20, suffix=''):
        print("Converting current photo to SVG")
        if os.path.isfile(image_filepath):
            image = cv2.imread(image_filepath)
            if image is not None:
                # Detect faces using dlib
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray_image)

                if len(faces) > 0:
                    # Use the largest detected face
                    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                    image = self.crop_to_largest_face(image, largest_face, target_width, target_height)
                else:
                    opt_image = image
                    print("No face found. Proceeding with the original image.")
                
                # Optimize the image
                opt_image = self.process_face_image(image)
                optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized.' + image_filepath.rsplit('.', 1)[1]
                cv2.imwrite(optimized_image_path, opt_image)
                
                # Convert the optimized image to SVG
                gray_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_image, threshold1=80, threshold2=200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
                simplified_contours = [cv2.approxPolyDP(contour, 1, True) for contour in filtered_contours]

                # Optimization loop for paths
                num_paths = 0
                trials = 0
                lower_threshold = 80
                upper_threshold = 200
                epsilon = 1

                while num_paths < min_paths or num_paths > max_paths:
                    dwg = svgwrite.Drawing(size=(target_width, target_height))
                    num_paths = self.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)

                    if num_paths < min_paths or num_paths > max_paths:
                        lower_threshold, upper_threshold, epsilon = self.adjust_thresholds(
                            num_paths, min_paths, max_paths, lower_threshold, upper_threshold, epsilon, trials)
                        edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        simplified_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

                    trials += 1
                    if trials > self.maxTrials:
                        break

                # Sort contours by distance from center and limit paths if necessary
                image_center = np.array([target_width // 2, target_height // 2])
                sorted_contours = sorted(simplified_contours, key=lambda c: np.linalg.norm(np.mean(np.squeeze(c), axis=0) - image_center))

                simplified_contours = sorted_contours[:max_paths] if len(sorted_contours) > max_paths else sorted_contours
                dwg = svgwrite.Drawing(size=(target_width, target_height))
                self.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)

                # Save the SVG
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(parent_dir, "photos/traced")
                os.makedirs(output_dir, exist_ok=True)
                
                svg_filename = os.path.splitext(os.path.basename(image_filepath))[0] + suffix + '.svg'
                svg_filepath = os.path.join(output_dir, svg_filename)
                dwg.saveas(svg_filepath)
                
                return svg_filepath

        return None

    def create_output_svg(self, image_svg_path, imgname = 'image', scale_factor = 0.8, offset_x=0, offset_y=0, id=0):
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
                if points:
                    points_tuples = re.findall(r'(-?\d*\.?\d+)[,\s](-?\d*\.?\d+)', points)
                    if points_tuples:
                        group.add(dwg.polyline(points=points_tuples, 
                                            stroke=element.get('stroke', 'black'),
                                            fill=element.get('fill', 'none'),
                                            stroke_width=element.get('stroke-width', '1')))

        dwg.add(group)
        
         # Use an absolute path for the output directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "photos/current")  # Join it with your relative path
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # svg_filename = os.path.splitext(os.path.basename(image_svg_path))[0] + str(id) + '.svg'
        svg_filename = imgname + str(id) + '.svg'
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