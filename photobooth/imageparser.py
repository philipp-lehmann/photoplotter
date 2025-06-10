import os
import re
import random

import cv2
import dlib
import numpy as np
import svgwrite
import torch
from torchvision import transforms
from scipy.spatial import cKDTree
from lxml import etree
from utils import profile

class ImageParser:
    def __init__(self):
        print("Starting ImageParser ...")

        # Initialize dlib face detector and shape predictor
        self.face_detector = dlib.get_frontal_face_detector()
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Loading 68 face landmarks model
        predictor_path = os.path.join(base_path, 'shape_predictor', 'shape_predictor_68_face_landmarks.dat')
        self.landmark_detector = dlib.shape_predictor(predictor_path)

        # Initialize MiDaS model for depth estimation
        model_type = "MiDaS_small"
        model_path = os.path.join(base_path, 'midas', 'midas_small.pth')

        if not os.path.exists(model_path):
            print("Model not found locally. Downloading...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
        else:
            print("Loading model from local path...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((192, 192)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def detect_faces(self, image_filepath):
        """Used when snapping an image. Quick method to check if a face is present in the image"""
        image = cv2.imread(image_filepath)
        if image is None:
            print("Failed to load image.")
            return False
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image)
        return len(faces) > 0
    
    def convert_to_svg(self, image_filepath, target_width=800, target_height=800, scale_x=1.0, scale_y=1.0, min_paths=30, max_paths=250, min_contour_area=20, suffix='', method=1):
        """Convert input image to SVG with parameters."""
        print(f"Converting {image_filepath}")
        if not os.path.isfile(image_filepath):
            print(f"File {image_filepath} does not exist.")
            return None
        
        image = cv2.imread(image_filepath)
        if image is None:
            print("Image loading failed.")
            return None
        
        image = self.handle_faces(image, target_width, target_height)
        opt_image = self.process_face_image(image)
        opt_image, depth_map = self.generate_and_apply_depth_map(image, opt_image)
        self.save_optimized_image(image_filepath, opt_image, depth_map)
        
        # Extract contours from the optimized image
        image_contours = self.extract_contours(opt_image, method, min_contour_area)
        image_contours = self.sort_and_limit_contours(image_contours, target_width, target_height, max_paths)
        
        # Extract contours from the depth map
        depth_map_contours = self.extract_contours(depth_map, method, min_contour_area)
        depth_map_contours = self.sort_and_limit_contours(depth_map_contours, target_width, target_height, max_paths)
        
        # Merge contours
        merged_contours = image_contours + depth_map_contours
        
        # Create and process the SVG
        svg_filepath = self.create_svg(image_filepath, merged_contours, target_width, target_height, scale_x, scale_y, suffix)
        processed_svg_filepath = self.process_svg(svg_filepath, depth_map)
        
        print(f"Processed SVG saved at: {processed_svg_filepath}")
        return processed_svg_filepath
    
    
    # ----- Face Detection -----
    @profile
    def handle_faces(self, image, target_width, target_height):
        """Detect and crop faces from the image or return the original."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image)
        if faces:
            return self.crop_all_faces(image, faces, target_width, target_height)
        print("No face found. Proceeding with the original image.")
        return image
    
    @profile
    def crop_all_faces(self, image, faces, target_width=400, target_height=400, padding=350):
        """Crop the image to a bounding rectangle encompassing all faces and resize it."""
        
        # Check if 'faces' is a single rectangle or a collection of rectangles and print detected faces
        if isinstance(faces, dlib.rectangle):
            faces = [faces]  # Wrap it in a list
        elif not isinstance(faces, (dlib.rectangles, list)):
            raise TypeError("'faces' must be a dlib.rectangle or dlib.rectangles object.")

        print("Detected faces:")
        for i, face in enumerate(faces):
            print(f"Face {i}: Left={face.left()}, Top={face.top()}, Right={face.right()}, Bottom={face.bottom()}")

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
            padding_top = diff // 2
            padding_bottom = diff - padding_top
            min_y = max(0, min_y - padding_top)
            max_y = min(image.shape[0], max_y + padding_bottom)
        elif height > width:
            diff = height - width
            padding_left = diff // 2
            padding_right = diff - padding_left
            min_x = max(0, min_x - padding_left)
            max_x = min(image.shape[1], max_x + padding_right)

        # Ensure the final bounding box is within image bounds
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(image.shape[1], max_x), min(image.shape[0], max_y)

        # Verify the bounding box is square
        width = max_x - min_x
        height = max_y - min_y
        # assert width == height, f"Bounding box must be square. Width: {width}, Height: {height}"
        
        # Crop the image
        cropped_image = image[min_y:max_y, min_x:max_x]
        
        # Resize the cropped image to the target size
        resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized_image
    
    @profile
    def process_face_image(self, image, target_width=800, target_height=800):
        """Optimized method that detects, crops, enhances the face and draws facial features"""
        if image is None:
            print("Failed to load image.")
            return None
        
        # Detect faces
        opt_image = cv2.medianBlur(image, 5)
        gray_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2GRAY)
        return gray_image


    # ----- Depth Map -----
    @profile
    def generate_and_apply_depth_map(self, image, opt_image):
        """Generate a depth map and apply it to the optimized image."""
        depth_map = self.generate_depth_map(image)
        depth_map = cv2.resize(depth_map, (opt_image.shape[1], opt_image.shape[0]))
        
        if depth_map is not None:
            opt_image = self.enhance_foreground(opt_image, depth_map)
            return opt_image, depth_map
        else:
            print("Depth map generation failed, proceeding without enhancement.")
            return opt_image, None  
        
    def generate_depth_map(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            depth_map = self.model(input_tensor)

        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
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

    # ----- Save image -----     
    def save_optimized_image(self, image_filepath, opt_image, depth_map=None):
        """Save the optimized image and optionally the depth map, and return the paths."""
        # Save the optimized image
        optimized_image_path = image_filepath.rsplit('.', 1)[0] + '_optimized.' + image_filepath.rsplit('.', 1)[1]
        cv2.imwrite(optimized_image_path, opt_image)
        
        # Save the depth map if it exists
        if depth_map is not None:
            depth_map_path = image_filepath.rsplit('.', 1)[0] + '_depthmap.png'
            cv2.imwrite(depth_map_path, depth_map)
            return optimized_image_path, depth_map_path
        
        return optimized_image_path, None

    # ----- Contours -----  
    def extract_contours(self, opt_image, method, min_contour_area):
        """Extract contours based on the selected method."""
        contours = []
        if method == 1:  # Edge-based
            edges = cv2.Canny(opt_image, threshold1=80, threshold2=200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == 2:  # Binary-based
            _, binary = cv2.threshold(opt_image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif method == 3:  # Merge both
            edges = cv2.Canny(opt_image, threshold1=80, threshold2=200)
            edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _, binary = cv2.threshold(opt_image, 127, 255, cv2.THRESH_BINARY)
            binary_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = edge_contours + binary_contours
        return [c for c in contours if cv2.contourArea(c) > min_contour_area]

    def sort_and_limit_contours(self, contours, target_width, target_height, max_paths):
        """Sort and limit the number of contours to a specified maximum."""
        image_center = np.array([target_width // 2, target_height // 2])
        sorted_contours = sorted(contours, key=lambda c: np.linalg.norm(np.mean(np.squeeze(c), axis=0) - image_center))
        return sorted_contours[:max_paths]

    # ----- SVG Handling -----  
    def create_svg(self, image_filepath, contours, target_width, target_height, scale_x, scale_y, suffix):
        """Create an SVG file from contours."""
        dwg = svgwrite.Drawing(size=(target_width, target_height))
        self.add_contours_to_svg(dwg, contours, scale_x, scale_y)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "photos/traced")
        os.makedirs(output_dir, exist_ok=True)
        svg_filename = os.path.splitext(os.path.basename(image_filepath))[0] + suffix + '.svg'
        svg_filepath = os.path.join(output_dir, svg_filename)
        dwg.saveas(svg_filepath)
        return svg_filepath
    
    
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

    # ----- SVG Processing -----  
    @profile
    def process_svg(self, svg_filepath, depth_map):
        """Process the SVG file to align it to points generated by various methods."""
        
        # Set the point generation methed here...
        grid_points = self.generate_dynamic_points()

        # method = random.choice(["dynamic_grid", "poisson_disk", "depth_map_density", "quad_tree"])
        # if method == "dynamic_grid":
        #     print("Using Dynamic Grid method for point generation.")
        #   grid_points = self.generate_dynamic_points()
        # elif method == "poisson_disk":
        #     print("Using Poisson Disk Sampling method for point generation.")
        #     grid_points = self.generate_poisson_disk_points()
        # elif method == "quad_tree":
        #     print("Using Quad Tree method for point generation.")
        #     grid_points = self.generate_quad_tree_points()
        # else:
        # print("Using Depth Map Density method for point generation.")
        # grid_points = self.generate_depth_based_points(depth_map)
        
        aligned_svg_filepath = svg_filepath.rsplit('.', 1)[0] + '_aligned.svg'
        self.align_svg_to_points(svg_filepath, aligned_svg_filepath, grid_points)
        
        cleaned_svg_filepath = self.remove_duplicate_segments(aligned_svg_filepath)
        return cleaned_svg_filepath

    def generate_dynamic_points(self, min_value=0, max_value=800, num_points_mean=70, num_points_std=10, center=400.0, plateau_radius=50, randomness_factor=0.1):
        """Generate a dynamic grid as an array of (x, y) points with higher density near the center,
        a plateau region, and increasing randomness towards the edges.

        num_points is sampled from a Gaussian distribution clipped between 40 and 100.
        """
        # Sample num_points from a Gaussian distribution and clip between 40 and 100
        num_points = int(np.clip(np.random.normal(loc=num_points_mean, scale=num_points_std), 40, 100))
        print(f"Generated grid with \033[1;31m{num_points}\033[0m points.")

        # Generate cubic-scaled values for x and y
        values = np.linspace(-1, 1, num_points)
        scaled_values = center + (max_value - min_value) * values**3

        grid_points = []
        for x in scaled_values:
            for y in scaled_values:
                # Calculate the distance from the center
                distance = ((x - center)**2 + (y - center)**2)**0.5

                # Keep points within the plateau radius
                if distance <= plateau_radius:
                    grid_points.append((x, y))
                else:
                    # Introduce randomness for points outside the plateau radius
                    edge_factor = min(1.0, distance / (max_value - min_value))  # Normalize edge factor to [0, 1]
                    if random.random() > randomness_factor * edge_factor:
                        grid_points.append((x, y))

        return grid_points
        
    def generate_quad_tree_points(self, min_value=0, max_value=800, min_square_size=2, max_square_size=160, center=(400.0, 400.0), grid_resolution=20):
        """Generate points based on a quad tree-like structure, with smaller squares in the center
        and larger squares toward the edges, ensuring regular grid distribution."""
        
        # List to store generated points
        points = []

        # Determine the total area size (x and y ranges)
        total_size = max_value - min_value

        # Create a grid with specified resolution (spacing between points)
        for x in range(min_value, max_value, grid_resolution):
            for y in range(min_value, max_value, grid_resolution):
                # Compute the distance from the center of the grid
                dist_to_center = ((x - center[0])**2 + (y - center[1])**2)**0.5

                # Determine the square size based on the distance to the center
                square_size = min_square_size + (max_square_size - min_square_size) * (dist_to_center / total_size)
                square_size = min(max_square_size, max(min_square_size, square_size))  # Clamp to min/max size
                
                # If this grid point is within a square's bounds, add it as a point
                if dist_to_center <= square_size:
                    points.append((x, y))

        return points
    
    def generate_quad_tree_points(self, min_value=0, max_value=800, min_square_size=10, max_square_size=100, center=(400.0, 400.0)):
        """Generate points based on a quad tree-like structure, with smaller squares in the center
        and larger squares toward the edges."""
        
        def subdivide_area(x_min, y_min, x_max, y_max, level):
            """Recursively subdivide the area to generate points."""
            # Compute the center of the current area
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Compute the distance from the center of the quad to the global center
            dist_to_center = ((x_center - center[0])**2 + (y_center - center[1])**2)**0.5
            
            # Determine the size of the current square
            square_size = min_square_size + (max_square_size - min_square_size) * (dist_to_center / (max_value - min_value))
            square_size = min(max_square_size, max(min_square_size, square_size))  # Clamp to min/max size
            
            # If the current square is sufficiently small, stop subdividing and add the point
            if (x_max - x_min) <= square_size or (y_max - y_min) <= square_size:
                points.append((x_center, y_center))
                return
            
            # Otherwise, subdivide further
            subdivide_area(x_min, y_min, x_center, y_center, level + 1)  # Top-left
            subdivide_area(x_center, y_min, x_max, y_center, level + 1)  # Top-right
            subdivide_area(x_min, y_center, x_center, y_max, level + 1)  # Bottom-left
            subdivide_area(x_center, y_center, x_max, y_max, level + 1)  # Bottom-right
        
        # Initialize the list of points
        points = []
        
        # Start subdivision from the entire area
        subdivide_area(min_value, min_value, max_value, max_value, level=0)
        
        return points

    def generate_poisson_disk_points(self, width=800, height=800, radius=20, k=30):
        """Generate points using Poisson Disk Sampling."""
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        grid_size = radius / np.sqrt(2)
        cols, rows = int(width // grid_size), int(height // grid_size)
        grid = [None] * (cols * rows)
        
        def grid_index(x, y):
            col = int(x // grid_size)
            row = int(y // grid_size)
            if 0 <= col < cols and 0 <= row < rows:
                return col + row * cols
            else:
                return -1 

        points = []
        active_list = []

        def add_point(x, y):
            idx = grid_index(x, y)
            if idx != -1:
                grid[idx] = (x, y)
                points.append((x, y))
                active_list.append((x, y))

        add_point(random.uniform(0, width), random.uniform(0, height))

        while active_list:
            x, y = active_list.pop(random.randint(0, len(active_list) - 1))
            for _ in range(k):
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(radius, 2 * radius)
                nx, ny = x + r * np.cos(angle), y + r * np.sin(angle)
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                neighbor_found = False
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        idx = grid_index(nx + i * grid_size, ny + j * grid_size)
                        if 0 <= idx < len(grid) and grid[idx] and distance(grid[idx], (nx, ny)) < radius:
                            neighbor_found = True
                            break
                    if neighbor_found:
                        break
                if not neighbor_found:
                    add_point(nx, ny)
        return points
    
    def generate_depth_based_points(self, depth_map, target_points=2000, min_points_per_region=1, max_points_per_region=10):
        """Generate points with density based on the depth map."""
        height, width = depth_map.shape
        points = []
        radius_map = cv2.normalize(1 / (depth_map + 1e-5), None, 5, 50, cv2.NORM_MINMAX)  # Inverse depth for density

        # Generate points with depth-based density
        for y in range(height):
            for x in range(width):
                radius = int(radius_map[y, x])
                num_points = int(np.interp(radius, [5, 50], [max_points_per_region, min_points_per_region]))
                
                for _ in range(num_points):
                    offset_x = random.uniform(-radius, radius)
                    offset_y = random.uniform(-radius, radius)
                    if offset_x**2 + offset_y**2 <= radius**2:  # Stay within the circular region
                        new_x = x + offset_x
                        new_y = y + offset_y
                        if 0 <= new_x < width and 0 <= new_y < height:
                            points.append((new_x, new_y))

        # Normalize total number of points to the target
        if len(points) > target_points:
            points = random.sample(points, target_points)

        print(f"Generated {len(points)} points based on depth map (target was {target_points}).")
        return points
    
    
    # ----- SVG Align and Cleanup -----  
    @profile
    def align_svg_to_points(self, input_file, output_file, grid_points):
        """Process SVG by aligning points to a dynamic grid using lxml.etree."""
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(input_file, parser)
        root = tree.getroot()

        # Build KD-Tree for faster nearest-point lookup
        kdtree = self.precompute_kdtree(grid_points)

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        for poly in root.xpath(".//svg:polyline | .//svg:polygon", namespaces=namespace):
            points = poly.get("points")
            if points:
                new_points = []
                for point in points.split():
                    try:
                        x, y = map(float, point.split(","))
                        aligned_point = self.align_to_dynamic_grid((x, y), kdtree)
                        new_points.append(f"{aligned_point[0]},{aligned_point[1]}")
                    except ValueError:
                        continue  # Skip malformed points
                poly.set("points", " ".join(new_points))

        tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        print(f"Aligned SVG saved as {output_file}")
        return output_file
    
    def precompute_kdtree(self, grid_points):
        return cKDTree(grid_points)

    def align_to_dynamic_grid(self, point, kdtree):
        _, idx = kdtree.query(point)
        return kdtree.data[idx]

    @profile
    def remove_duplicate_segments(self, svg_filepath):
        """Remove duplicate line segments from polylines in an SVG file."""
        import xml.etree.ElementTree as ET

        def parse_points(points_str):
            """Parse the points attribute into a list of (x, y) tuples."""
            points = points_str.strip().split()
            return [tuple(map(float, point.split(','))) for point in points]

        def format_points(points):
            """Format a list of (x, y) tuples into a points attribute string."""
            return ' '.join(f"{x},{y}" for x, y in points)

        def get_line_key(point1, point2):
            """Generate a unique key for a line segment based on its endpoints."""
            return tuple(sorted([point1, point2]))

        tree = ET.parse(svg_filepath)
        root = tree.getroot()

        seen_segments = set()
        new_polylines = []

        for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
            points = parse_points(polyline.attrib.get('points', ''))
            unique_points = []

            for i in range(len(points) - 1):
                segment_key = get_line_key(points[i], points[i + 1])
                if segment_key not in seen_segments:
                    seen_segments.add(segment_key)
                    if not unique_points or unique_points[-1] != points[i]:
                        unique_points.append(points[i])
                    unique_points.append(points[i + 1])

            if unique_points:
                # Avoid duplicates at the start of the next polyline
                unique_points = [unique_points[0]] + [
                    pt for i, pt in enumerate(unique_points[1:], start=1)
                    if pt != unique_points[i - 1]
                ]
                new_polylines.append((polyline, format_points(unique_points)))

        # Update polylines in the tree
        for polyline, new_points in new_polylines:
            polyline.set('points', new_points)

        # Save the cleaned SVG
        tree.write(svg_filepath)
        return svg_filepath

    
    # ----- SVG Output -----
    @profile  
    def create_output_svg(self, image_svg_path, imgname = 'image', scale_factor = 0.33, offset_x=0, offset_y=0, id=0):
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
        output_dir = os.path.join(parent_dir, "photos/output")  # Join it with your relative path
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # svg_filename = os.path.splitext(os.path.basename(image_svg_path))[0] + str(id) + '.svg'
        svg_filename = imgname + str(id) + '.svg'
        output_svg_path = os.path.join(output_dir, svg_filename)  # This is your absolute path for the SVG file
        dwg.saveas(output_svg_path)

        return output_svg_path
    
    # ----- SVG Collection -----  
    def collect_all_paths(self, input_directory, output_file):
        """Combines all SVG files in the input directory. """
        # Get all SVG files in the directory, sorted alphabetically
        svg_files = sorted(f for f in os.listdir(input_directory) if f.endswith('.svg'))
        if not svg_files:
            print("No SVG files found in the directory.")
            return

        # Parse the first file to use its <svg> structure
        first_svg_path = os.path.join(input_directory, svg_files[0])
        with open(first_svg_path, 'rb') as file:
            first_svg_data = file.read()
        first_root = etree.fromstring(first_svg_data)
        namespace = {'svg': "http://www.w3.org/2000/svg"}
        
        # Remove any content inside the first <svg> tag (except attributes)
        for child in list(first_root):
            first_root.remove(child)

        # Append contents from all files
        for filename in svg_files:
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'rb') as file:
                svg_data = file.read()
            root = etree.fromstring(svg_data)

            # Append all children of the current SVG (excluding the outer <svg> tag)
            for child in root:
                if child.tag.endswith('g'):  # Handle <g> tags explicitly
                    if len(child):  # Check if the <g> tag has children
                        first_root.append(child)
                    else:
                        # If <g> is self-closing, convert to open-close format
                        g = etree.Element('g', attrib=child.attrib)
                        first_root.append(g)
                else:
                    first_root.append(child)

        # Save the combined SVG to the output file
        with open(output_file, 'wb') as file:
            file.write(etree.tostring(first_root, pretty_print=True))

        print(f"Combined SVG saved to {output_file}")   
    
    # ----- NOT IN USE -----
    def crop_to_largest_face(self, image, face_rect, target_width=400, target_height=400):
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
        blended_image = cv2.addWeighted(image, 0.8, enhanced_image, 0.2, 0)
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
        probability_eyebrows = 0.24 
        probability_mouth = 0.22 
        probability_jawline = 0.27
        probability_eyes_1 = 0.31
        probability_eyes_2 = 0.35
        probability_nose_1 = 0.34
        probability_nose_2 = 0.21
        probability_teeth = 0.17

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