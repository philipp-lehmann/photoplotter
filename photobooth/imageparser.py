import os
import re
import random

import cv2
import dlib
import numpy as np
import svgwrite
import torch
from torchvision import transforms
from lxml import etree

class ImageParser:
    def __init__(self):
        self.maxTrials = 10
        print("Starting ImageParser ...")

        # Initialize dlib face detector and shape predictor (68 landmarks)
        self.face_detector = dlib.get_frontal_face_detector()
        base_path = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(base_path, 'shape_predictor', 'shape_predictor_68_face_landmarks.dat')

        # Validate file paths
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}")
        
        # Load shape predictor
        self.landmark_detector = dlib.shape_predictor(predictor_path)     
        print("Starting ImageParser ...")
        pass    
    
    
    def process_face_image(self, image, target_width=800, target_height=800):
        """Optimized method that detects, crops, enhances the face and draws facial features"""
        if image is None:
            print("Failed to load image.")
            return None
        
        # Detect faces
        opt_image = cv2.medianBlur(image, 5)
        gray_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2GRAY)
        return gray_image


    def crop_all_faces(self, image, faces, target_width=800, target_height=800, padding=1500):
        """Crop the image to a bounding rectangle encompassing all faces and resize it."""
        # Check if 'faces' is a single rectangle or a collection of rectangles
        if isinstance(faces, dlib.rectangle):
            faces = [faces]  # Wrap it in a list
        elif not isinstance(faces, (dlib.rectangles, list)):
            raise TypeError("'faces' must be a dlib.rectangle or dlib.rectangles object.")

        # Initialize bounding box coordinates
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        # Calculate the encompassing bounding box
        for face in faces:
            min_x = min(min_x, face.left())
            min_y = min(min_y, face.top())
            max_x = max(max_x, face.right())
            max_y = max(max_y, face.bottom())
        
        # Add padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[1], max_x + padding)
        max_y = min(image.shape[0], max_y + padding)
        
        # Adjust bounding box to a square
        width = max_x - min_x
        height = max_y - min_y
        
        if width > height:
            diff = width - height
            min_y = max(0, min_y - diff // 2)
            max_y = min(image.shape[0], max_y + diff - (min_y == 0) * diff)
        elif height > width:
            diff = height - width
            min_x = max(0, min_x - diff // 2)
            max_x = min(image.shape[1], max_x + diff - (min_x == 0) * diff)
        
        # Final bounding box dimensions
        width = max_x - min_x
        height = max_y - min_y
        
        assert width == height, "Bounding box must be square."
        
        # Crop the image
        cropped_image = image[min_y:max_y, min_x:max_x]
        
        # Resize the cropped image to the target size
        resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized_image

    def crop_to_largest_face(self, image, face_rect, target_width=800, target_height=800):
        """
        NOT IN USE: This function is no longer active.
        Crop the image around the detected face to a square size.
        """
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        center_x, center_y = x + w // 2, y + h // 2

        # Determine the size of the square crop
        crop_size = max(w, h)
        margin = int(crop_size * 0.14)  # Add some margin around the face
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
        """
        NOT IN USE: This function is no longer active.
        Increase the contrast of the input image
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_l_channel = clahe.apply(l_channel)
        enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
        blended_image = cv2.addWeighted(image, 0.65, enhanced_image, 0.35, 0)
        return blended_image

    def draw_facial_landmarks(self, image, face_rect):
        """
        NOT IN USE: This function is no longer active.
        Draw the 68 facial landmarks using dlib's shape predictor.
        """
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
            
    def generate_depth_map(self, image):
        """Generate a depth map for the image."""
        # Load the pre-trained depth estimation model (MiDaS or EfficientMon)
        # Example: Use a pre-trained MiDaS model or EfficientMon here
        model_type = "MiDaS_small"
        model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image_rgb).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            depth_map = model(input_tensor)

        depth_map = depth_map.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255
        return np.uint8(depth_map_normalized)
    
    def enhance_foreground(self, image, depth_map, contrast_factor=2.0, background_factor=0.5):
        """
        Enhance the foreground of the image based on a depth map.
        The foreground contrast is enhanced, and the background is darkened.

        Parameters:
            image (ndarray): The original grayscale image.
            depth_map (ndarray): The depth map of the image.
            contrast_factor (float): Factor to enhance contrast in the foreground.
            background_factor (float): Factor to darken the background.

        Returns:
            result (ndarray): The processed image with enhanced foreground and darkened background.
        """
        # Convert the depth map to a float and normalize to [0, 1]
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        # Create a mask for the foreground (near objects, typically with lower depth values)
        foreground_mask = (depth_map_normalized >= 0.5).astype(np.uint8) 
        background_mask = (depth_map_normalized < 0.5).astype(np.uint8) 

        # Enhance foreground contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)  # Applying CLAHE on grayscale image directly

        # Darken the background: Reduce the brightness of background pixels
        image_background_darker = image * background_factor  # Dim the background

        # Ensure the image background is clipped to 0-255 range (if necessary)
        image_background_darker = np.clip(image_background_darker, 0, 255).astype(np.uint8)

        # Combine the results: Blend the enhanced foreground and darkened background
        result = np.zeros_like(image, dtype=np.uint8)

        # Apply enhanced foreground to the result where the foreground mask is 1
        result[foreground_mask == 1] = enhanced_image[foreground_mask == 1]

        # Apply darkened background to the result where the background mask is 1
        result[background_mask == 1] = image_background_darker[background_mask == 1]

        return result


    def convert_to_svg(self, image_filepath, target_width=800, target_height=800, scale_x=0.35, scale_y=0.35, min_paths=30, max_paths=300, min_contour_area=20, suffix='', method=1):
        """Convert input image to svg with parameters."""
        print(f"Converting {image_filepath}")
        if os.path.isfile(image_filepath):
            image = cv2.imread(image_filepath)
            if image is not None:
                # Detect faces using dlib
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray_image)

                if len(faces) > 0:
                    # Use the largest detected face
                    image = self.crop_all_faces(image, faces, target_width, target_height)
                else:
                    print("No face found. Proceeding with the original image.")
                
                # Optimize the image
                opt_image = self.process_face_image(image)
                # optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized.' + image_filepath.rsplit('.', 1)[1]
                # cv2.imwrite(optimized_image_path, opt_image)
                
                # Generate depth map and integrate it with the image
                depth_map = self.generate_depth_map(image)
                depth_map = cv2.resize(depth_map, (opt_image.shape[1], opt_image.shape[0]))

                # Apply the depth map to the optimized image
                opt_image = self.enhance_foreground(opt_image, depth_map)
                optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized.' + image_filepath.rsplit('.', 1)[1]
                cv2.imwrite(optimized_image_path, opt_image)
                
                # Initialize empty list for contours
                contours = []
                
                

                # Extract contours based on the selected method
                if method == 1:  # Edge-based method
                    edges = cv2.Canny(opt_image, threshold1=80, threshold2=200)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                elif method == 2:  # Binary-based method
                    _, binary = cv2.threshold(opt_image, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                elif method == 3:  # Merge both methods
                    edges = cv2.Canny(opt_image, threshold1=80, threshold2=200)
                    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    _, binary = cv2.threshold(opt_image, 127, 255, cv2.THRESH_BINARY)
                    binary_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = edge_contours + binary_contours

                # Filter and simplify contours
                filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
                simplified_contours = [cv2.approxPolyDP(c, 1, True) for c in filtered_contours]
                image_center = np.array([target_width // 2, target_height // 2])
                sorted_contours = sorted(simplified_contours, key=lambda c: np.linalg.norm(np.mean(np.squeeze(c), axis=0) - image_center))
                simplified_contours = sorted_contours[:max_paths]

                # Create SVG
                dwg = svgwrite.Drawing(size=(target_width, target_height))
                self.add_contours_to_svg(dwg, simplified_contours, scale_x, scale_y)
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(parent_dir, "photos/traced")
                os.makedirs(output_dir, exist_ok=True)
                svg_filename = os.path.splitext(os.path.basename(image_filepath))[0] + suffix + '.svg'
                svg_filepath = os.path.join(output_dir, svg_filename)
                dwg.saveas(svg_filepath)

                # Process the SVG
                x_grid = self.generate_dynamic_grid()
                y_grid = self.generate_dynamic_grid()
                processed_svg_filepath = svg_filepath.rsplit('.', 1)[0] + '_aligned.svg'
                self.align_svg_to_points(svg_filepath, processed_svg_filepath, x_grid, y_grid)

                print(f"Processed SVG saved at: {processed_svg_filepath}")
                return processed_svg_filepath

        return None
    
    def generate_dynamic_grid(self, min_value=0, max_value=280, num_points=70, center=140.0):
        """Generate a dynamic grid with higher density near the center and map values to the specified range."""
        
        # Generate original grid values (e.g., cubic scaling)
        values = np.linspace(-1, 1, num_points)
        original_dynamic_values = center + (max_value - min_value) * values**3
        
        # Calculate the min and max of the original dynamic grid values
        original_min = np.min(original_dynamic_values)
        original_max = np.max(original_dynamic_values)
        
        # Map original values to the target range [min_value, max_value]
        mapped_dynamic_values = [
            min_value + (val - original_min) / (original_max - original_min) * (max_value - min_value)
            for val in original_dynamic_values
        ]        
        return sorted(mapped_dynamic_values)
    
    
    @staticmethod
    def align_to_dynamic_grid(value, grid):
        """Find the closest value in the dynamic grid."""
        return min(grid, key=lambda x: abs(x - value))

    
    def align_svg_to_points(self, input_file, output_file, x_grid, y_grid):
        """Process SVG by aligning points to a dynamic grid using lxml.etree."""
        # Parse the SVG file
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(input_file, parser)
        root = tree.getroot()

        # Define the SVG namespace
        namespace = {'svg': 'http://www.w3.org/2000/svg'}

        # Process all polyline and polygon elements
        for poly in root.xpath(".//svg:polyline | .//svg:polygon", namespaces=namespace):
            points = poly.get("points")
            if points:
                new_points = []
                for point in points.split():
                    try:
                        x, y = map(float, point.split(","))
                        x_aligned = self.align_to_dynamic_grid(x, x_grid)
                        y_aligned = self.align_to_dynamic_grid(y, y_grid)
                        new_points.append(f"{x_aligned},{y_aligned}")
                    except ValueError:
                        continue  # Skip malformed points
                poly.set("points", " ".join(new_points))

        # Save the modified SVG file
        tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        print(f"Processed SVG saved as {output_file}")
        return output_file

    def create_output_svg(self, image_svg_path, imgname = 'image', scale_factor = 0.8, offset_x=0, offset_y=0, id=0):
        """Create output image on artboard with id for output position"""
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