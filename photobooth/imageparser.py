import cv2
import os
import dlib
import numpy as np
import svgwrite
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
        # Load the pre-trained Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load the image
        image = cv2.imread(image_filepath)
        if image is None:
            print("Failed to load image.")
            return 0  # Return 0 as confidence if the image is not loaded
        
        # Convert the image to grayscale (required for face detection)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Check if any faces are detected
        if len(faces) == 0:
            return False
        else:
            print(f"Detected {len(faces)} face(s) in the image.")
            
            # Calculate confidence score based on the largest detected face area
            largest_face_area = max([w * h for (x, y, w, h) in faces])
            image_area = image.shape[0] * image.shape[1]
            
            # Confidence calculation (heuristic)
            confidence = (largest_face_area / image_area) * 100
            confidence = min(confidence, 100)
            
            if confidence < 1.0:
                return False
            else: 
                print(f"Confidence score: {confidence:.2f}")
                return True
            
    def detect_facial_features(self, image, face_rect):
        # Load Haar cascades for facial features
        base_path = os.path.dirname(os.path.abspath(__file__))
        eye_cascade_path = os.path.join(base_path, 'haarcascades', 'eye.xml')
        nose_cascade_path = os.path.join(base_path, 'haarcascades', 'nose.xml')
        mouth_cascade_path = os.path.join(base_path, 'haarcascades', 'mouth.xml')

        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
        mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

        if eye_cascade.empty():
            print("Error: Eye cascade file not found or cannot be loaded.")
        if nose_cascade.empty():
            print("Error: Nose cascade file not found or cannot be loaded.")
        if mouth_cascade.empty():
            print("Error: Mouth cascade file not found or cannot be loaded.")
        
        x, y, w, h = face_rect
        face_region = image[y:y+h, x:x+w]
        
        # Convert face region to grayscale
        gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes, nose, and mouth
        eyes = eye_cascade.detectMultiScale(gray_face_region, scaleFactor=1.1, minNeighbors=5)
        nose = nose_cascade.detectMultiScale(gray_face_region, scaleFactor=1.1, minNeighbors=5) if not nose_cascade.empty() else []
        mouth = mouth_cascade.detectMultiScale(gray_face_region, scaleFactor=1.1, minNeighbors=5) if not mouth_cascade.empty() else []
        
        return eyes, nose, mouth
    
    def crop_to_largest_face(self, image, target_width, target_height, margin_percentage=0.60):
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
         # Enhance details in the image
        enhanced_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.255)
        
        # Convert the enhanced image to grayscale for face detection
        enhanced_gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(enhanced_gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(2, 2))

        if len(faces) == 0:
            return None  # No faces detected

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        margin = int(max(w, h) * margin_percentage)

        eyes, nose, mouth = self.detect_facial_features(image, largest_face)

        # Create a mask for the face region
        face_mask = np.zeros_like(enhanced_gray_image)
        face_mask[y:y+h, x:x+w] = 200

        # Calculate the new crop region
        crop_size = max(w, h) + 2 * margin
        crop_size = min(crop_size, image.shape[0], image.shape[1])

        if len(eyes) > 0 or len(nose) > 0 or len(mouth) > 0:
            features_x = [x + ex for (ex, ey, ew, eh) in eyes] + [x + nx for (nx, ny, nw, nh) in nose] + [x + mx for (mx, my, mw, mh) in mouth]
            features_y = [y + ey for (ex, ey, ew, eh) in eyes] + [y + ny for (nx, ny, nw, nh) in nose] + [y + my for (mx, my, mw, mh) in mouth]
            x_center = int(np.mean(features_x))
            y_center = int(np.mean(features_y))

            x_start = max(x_center - crop_size // 2, 0)
            y_start = max(y_center - crop_size // 2, 0)
            x_end = min(x_start + crop_size, image.shape[1])
            y_end = min(y_start + crop_size, image.shape[0])

            if x_end - x_start < crop_size:
                x_start = max(image.shape[1] - crop_size, 0)
                x_end = image.shape[1]
            if y_end - y_start < crop_size:
                y_start = max(image.shape[0] - crop_size, 0)
                y_end = image.shape[0]
        else:
            x_start = max(x + w // 2 - crop_size // 2, 0)
            y_start = max(y + h // 2 - crop_size // 2, 0)
            x_end = min(x_start + crop_size, image.shape[1])
            y_end = min(y_start + crop_size, image.shape[0])

        cropped_image = image[y_start:y_end, x_start:x_end]
        return cv2.resize(cropped_image, (target_width, target_height))
     
     
        
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

    def bilateral_filter_image(self, image, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


        
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

    def bilateral_filter_image(self, image, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    def optimize_image(self, img):
        # Detect faces using dlib
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        if len(faces) == 0:
            print("No faces detected")
            return img
        
        # For each detected face, find landmarks and draw them
        for face in faces:
            landmarks = self.landmark_detector(gray, face)
            
            # Separate facial regions (e.g., jawline, eyebrows, etc.)
            self.draw_feature_line(img, landmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            self.draw_feature_line(img, landmarks, [17, 18, 19, 20, 21])
            self.draw_feature_line(img, landmarks, [22, 23, 24, 25, 26])
            self.draw_feature_line(img, landmarks, [36, 37, 38, 39, 40, 41, 36])  # Left eye
            self.draw_feature_line(img, landmarks, [42, 43, 44, 45, 46, 47, 42])  # Right eye
            self.draw_feature_line(img, landmarks, [31, 32, 33, 34, 35])
            self.draw_feature_line(img, landmarks, [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48])  # Mouth outer
            self.draw_feature_line(img, landmarks, [60, 61, 62, 63, 64, 65, 66, 67, 60])  # Mouth inner

        # Return the image with the landmarks highlighted
        return img
    
    def draw_feature_line(self, img, landmarks, points_indices, color=(255, 255, 2550), thickness=1):
        points = []
        for i in points_indices:
            if 0 <= i < 68:  # Ensure valid indices
                points.append((landmarks.part(i).x, landmarks.part(i).y))
            else:
                print(f"Warning: Landmark index {i} is out of range and will be skipped.")

        # Draw lines connecting consecutive points
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, thickness)

    def convert_to_svg(self, image_filepath, target_width=800, target_height=800, scale_x=0.35, scale_y=0.35, min_paths=30, max_paths=90, min_contour_area=20, suffix=''):
        print("Converting current photo to SVG")
        if os.path.isfile(image_filepath):
            # Load the image using OpenCV
            image = cv2.imread(image_filepath)
            if image is not None:
                # Detect faces and crop around the largest face
                image = self.crop_to_largest_face(image, target_width, target_height)
                if image is None:
                    print("No face found. Proceeding with the original image.")
                    image = cv2.imread(image_filepath)  # Reload the original image if no face is found
                
                # Resize the image to the target dimensions
                image = cv2.resize(image, (target_width, target_height))
                
                # Optimize the image
                opt_image = self.optimize_image(image)
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