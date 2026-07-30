import os
import re
import random
import math
import urllib.request

import cv2
import dlib
import numpy as np
import svgwrite
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import cKDTree
from lxml import etree
from utils import profile, wait_for_cooldown, get_random_color, pc

SELFIE_SEGMENTER_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

# Canonical dlib 68-point landmark groups; eye/lip loops repeat their first index to close
LANDMARK_GROUPS = {
    "jaw": list(range(0, 17)),
    "right_brow": list(range(17, 22)),
    "left_brow": list(range(22, 27)),
    "nose_bridge": [27, 28, 29, 30],
    "nose_base": [31, 32, 33, 34, 35],
    "right_eye": [36, 37, 38, 39, 40, 41, 36],
    "left_eye": [42, 43, 44, 45, 46, 47, 42],
    "outer_lips": list(range(48, 60)) + [48],
    "inner_lips": list(range(60, 68)) + [60],
}

class ImageParser:
    def __init__(self):
        print("Starting ImageParser ...")

        # Initialize dlib face detector and shape predictor
        self.face_detector = dlib.get_frontal_face_detector()
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Loading 68 face landmarks model
        predictor_path = os.path.join(base_path, 'shape_predictor', 'shape_predictor_68_face_landmarks.dat')
        self.landmark_detector = dlib.shape_predictor(predictor_path)

        # Initialize MediaPipe selfie segmentation for person/background separation
        model_path = os.path.join(base_path, 'models', 'selfie_segmenter.tflite')
        if not os.path.exists(model_path):
            print("Segmentation model not found locally. Downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(SELFIE_SEGMENTER_URL, model_path)

        segmenter_options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_confidence_masks=True,
        )
        self.segmenter = mp_vision.ImageSegmenter.create_from_options(segmenter_options)
    
    def detect_faces(self, image_filepath):
        """Used when snapping an image. Quick method to check if a face is present in the image"""
        image = cv2.imread(image_filepath)
        if image is None:
            print("Failed to load image.")
            return False
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image)
        return len(faces) > 0
    
    def convert_to_svg(self, image_filepath, target_width=800, target_height=800, scale_x=1.0, scale_y=1.0, min_paths=30, max_paths=120, min_contour_area=16, suffix='', method=3, apply_depthmap=True, style=None, feature_radius=18, snap_method=None):
        """Convert input image to SVG with parameters.
        style: optional '+'-separated styles ('features', 'outline', 'oneline'), e.g. 'features+outline+oneline'.
        snap_method: force the point-snap style ('dynamic_grid'/'poisson_disk'/'none') instead of the random pick."""
        print(f"Converting {image_filepath}")
        if not os.path.isfile(image_filepath):
            print(f"File {image_filepath} does not exist.")
            return None
        
        image = cv2.imread(image_filepath)
        if image is None:
            print("Image loading failed.")
            return None
        
        styles = set((style or "").split("+"))
        crop_image = self.handle_faces(image, target_width, target_height)
        opt_image, faces, landmarks_list = self.process_face_image(crop_image)
        person_mask = None

        if apply_depthmap:
            opt_image, person_mask = self.generate_and_apply_mask(crop_image, opt_image)

        # Save optimized images
        self.save_optimized_image(image_filepath, opt_image, person_mask)

        # Occasionally swap Canny for XDoG sketch edges to vary the line style
        if method == 3 and random.random() < 0.3:
            method = 4

        # Extract contours from the optimized image
        image_contours = self.extract_contours(opt_image, method, min_contour_area, mask=person_mask)
        image_contours = self.sort_and_limit_contours(image_contours, target_width, target_height, max_paths, faces=faces)

        # Extract silhouette + offset "aura" rings from the person mask
        mask_contours = self.extract_mask_ring_contours(person_mask, min_contour_area)
        mask_contours = self.sort_and_limit_contours(mask_contours, target_width, target_height, max_paths, faces=faces)

        # Merge contours
        if "outline" in styles:
            # Outline style: keep the person silhouette as open paths without the
            # straight border-hugging segments; image contours only survive if the
            # features filter is also active (features+outline)
            if not mask_contours:
                print("No person silhouette found: outline style has nothing to draw.")
            mask_contours = self.split_contours_at_borders(mask_contours, target_width, target_height)
            if "features" in styles:
                image_contours = self.filter_contours_to_features(image_contours, landmarks_list, target_width, target_height, radius=feature_radius)
            else:
                image_contours = []
            merged_contours = image_contours + mask_contours
        else:
            merged_contours = image_contours + mask_contours
            # Keep only strokes overlapping the facial features (silhouette included,
            # so only its jaw-adjacent parts survive)
            if "features" in styles:
                merged_contours = self.filter_contours_to_features(merged_contours, landmarks_list, target_width, target_height, radius=feature_radius)

        # Create the SVG with a style
        svg_filepath = self.create_svg(image_filepath, merged_contours, target_width, target_height, scale_x, scale_y, suffix)
        if snap_method is not None:
            method = snap_method
        else:
            methods = ["dynamic_grid", "poisson_disk", "none"]
            weights = [0.025, 0.025, 0.95]
            method = random.choices(methods, weights=weights, k=1)[0]

        # Process SVG
        processed_svg_filepath = self.process_svg(svg_filepath, method)

        # Chain everything into one continuous line (after dedup/simplify, so the
        # single line is not chopped back apart by remove_duplicate_segments)
        if "oneline" in styles:
            processed_svg_filepath = self.chain_svg_polylines(processed_svg_filepath)

        print(f"{method}: Length Before {self.get_svgpath_length(svg_filepath)} / Output: {pc(self.get_svgpath_length(processed_svg_filepath))}")
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
        
        print("❌ No face found. Proceeding with the resized image.")
        return cv2.resize(image, (target_width, target_height))
    
    @profile
    def crop_all_faces(self, image, faces, target_width=800, target_height=800, padding=350):
        """Crop the image to a bounding rectangle encompassing all faces and resize it."""
        # Check if 'faces' is a single rectangle or a collection of rectangles and print detected faces
        if isinstance(faces, dlib.rectangle):
            faces = [faces]  # Wrap it in a list
        elif not isinstance(faces, (dlib.rectangles, list)):
            raise TypeError("'faces' must be a dlib.rectangle or dlib.rectangles object.")

        print("Detected faces:")
        for i, face in enumerate(faces):
            print(f"😄 \033[1;36mFace {i}: \033[0mLeft={face.left()}, Top={face.top()}, Right={face.right()}, Bottom={face.bottom()}")

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
        """Optimized method that detects, crops, enhances the face and draws facial features.
        Returns the cleaned grayscale image, the detected face rectangles and their landmarks."""
        if image is None:
            print("Failed to load image.")
            return None, [], []

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect at half resolution (HOG cost drops ~4x), scale rects back up
        small_gray = cv2.resize(gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        faces = [
            dlib.rectangle(f.left() * 2, f.top() * 2, f.right() * 2, f.bottom() * 2)
            for f in self.face_detector(small_gray)
        ]

        # Cartoon-style flattening: iterated edge-preserving smoothing flattens skin
        # and other low-contrast areas while keeping feature edges (eyes, mouth) sharp
        cleaned_image = gray_image
        for _ in range(2):
            cleaned_image = cv2.bilateralFilter(cleaned_image, 9, 40, 9)

        # Apply facial landmarks
        landmarks_list = []
        if faces:
            for face_rect in faces:
                landmarks = self.draw_facial_landmarks(cleaned_image, face_rect)
                if landmarks is not None:
                    landmarks_list.append(landmarks)

        return cleaned_image, faces, landmarks_list


    # ----- Person Mask (segmentation) -----
    @profile
    def generate_and_apply_mask(self, image, opt_image):
        """Generate a person mask and use it to enhance the foreground of the optimized image."""
        person_mask = self.generate_person_mask(image)

        if person_mask is not None:
            person_mask = cv2.resize(person_mask, (opt_image.shape[1], opt_image.shape[0]))
            opt_image = self.enhance_foreground(opt_image, person_mask)
            return opt_image, person_mask
        else:
            print("Person mask generation failed, proceeding without enhancement.")
            return opt_image, None

    def generate_person_mask(self, image):
        """Run MediaPipe selfie segmentation; returns a uint8 mask (255 = person)."""
        wait_for_cooldown()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.segmenter.segment(mp_image)

        if not result.confidence_masks:
            return None

        confidence = np.squeeze(result.confidence_masks[0].numpy_view())
        return np.uint8(np.clip(confidence * 255, 0, 255))
    
    def enhance_foreground(self, image, mask, contrast_factor=1.5, background_factor=0.5, feather=31):
        """
        Enhance the foreground contrast and darken the background, blended with a
        feathered mask. The soft transition avoids an artificial hard edge at the
        person's outline that edge detection would otherwise trace as extra paths.

        Parameters:
            image (ndarray): The original grayscale image.
            mask (ndarray): uint8 person mask (255 = person).
            contrast_factor (float): CLAHE clip limit for the foreground.
            background_factor (float): Factor to darken the background.
            feather (int): Gaussian kernel size for softening the mask edge (odd).

        Returns:
            result (ndarray): The processed image with enhanced foreground and darkened background.
        """
        alpha = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0

        clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
        enhanced = clahe.apply(image).astype(np.float32)
        background = image.astype(np.float32) * background_factor

        result = enhanced * alpha + background * (1.0 - alpha)
        return np.uint8(np.clip(result, 0, 255))


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
    def extract_contours(self, opt_image, method, min_contour_area, mask=None):
        """Extract contours based on the selected method.
        1 = auto-Canny edges, 2 = posterized shading iso-lines,
        3 = Canny + posterized (default), 4 = XDoG sketch + posterized."""
        if opt_image is None:
            return []
        contours = []
        if method == 1:
            contours = self.auto_canny_contours(opt_image)
        elif method == 2:
            contours = self.posterized_contours(opt_image, mask)
        elif method == 3:
            contours = self.auto_canny_contours(opt_image) + self.posterized_contours(opt_image, mask)
        elif method == 4:
            contours = self.xdog_contours(opt_image) + self.posterized_contours(opt_image, mask)

        filtered = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        # Remove pixel-staircase jitter so plotted lines are smooth
        return [cv2.approxPolyDP(c, 1.5, True) for c in filtered]

    def auto_canny_contours(self, image):
        """Canny edges with thresholds derived from the image median (robust to lighting)."""
        med = np.median(image)
        lower = int(max(0, 0.55 * med))
        upper = int(min(255, 1.25 * med))
        edges = cv2.Canny(image, lower, upper)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def posterized_contours(self, image, mask=None):
        """Topographic iso-lines: threshold the shading at 3-5 random levels and trace each.
        Restricted to the person's interior so the silhouette isn't re-traced per level."""
        num_levels = random.randint(4, 6)
        smooth = cv2.GaussianBlur(image, (3, 3), 0)

        interior = None
        if mask is not None:
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            interior = cv2.erode(mask_binary, erode_kernel)

        contours = []
        step = 256 // num_levels
        for level in range(step, 256, step):
            _, binary = cv2.threshold(smooth, level, 255, cv2.THRESH_BINARY)
            if interior is not None:
                binary = cv2.bitwise_and(binary, interior)
            level_contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours += level_contours

        if interior is not None:
            contours = self.drop_boundary_hugging_contours(contours, interior)
        return contours

    def drop_boundary_hugging_contours(self, contours, region_mask, band_width=7, max_fraction=0.5):
        """Discard contours that mostly trace the edge of the region mask instead of
        actual image features (masking per threshold level cuts every level's shape
        off at the mask edge, which would otherwise produce stacked outline paths)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width))
        band = cv2.morphologyEx(region_mask, cv2.MORPH_GRADIENT, kernel)
        h, w = band.shape
        kept = []
        for c in contours:
            pts = c.reshape(-1, 2)
            xs = np.clip(pts[:, 0], 0, w - 1)
            ys = np.clip(pts[:, 1], 0, h - 1)
            if (band[ys, xs] > 0).mean() < max_fraction:
                kept.append(c)
        return kept

    def xdog_contours(self, image, sigma=1.0, k=1.6, gamma=0.97, epsilon=-0.02, phi=15):
        """Extended Difference-of-Gaussians: hand-drawn-looking sketch strokes."""
        img = image.astype(np.float32) / 255.0
        g1 = cv2.GaussianBlur(img, (0, 0), sigma)
        g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
        dog = g1 - gamma * g2
        sketch = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
        sketch_u8 = np.uint8(np.clip(sketch * 255, 0, 255))
        _, strokes = cv2.threshold(sketch_u8, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(strokes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def extract_mask_ring_contours(self, mask, min_contour_area):
        """Single clean silhouette contour of the person from the segmentation mask."""
        if mask is None:
            return []
        _, silhouette = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        return [cv2.approxPolyDP(c, 1.5, True) for c in filtered]

    def split_contours_at_borders(self, contours, width, height, margin=3, min_run_length=10):
        """Split closed contours where they run along the image borders,
        returning open sub-contours without the border-hugging segments."""
        result = []
        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float64)
            on_border = (
                (pts[:, 0] <= margin) | (pts[:, 0] >= width - 1 - margin) |
                (pts[:, 1] <= margin) | (pts[:, 1] >= height - 1 - margin)
            )
            inside = ~on_border

            if inside.all():
                result.append(contour)
                continue
            if not inside.any():
                continue

            # Rotate so index 0 is on the border: no inside-run straddles the array boundary
            k = int(np.argmin(inside))
            inside = np.roll(inside, -k)
            pts = np.roll(pts, -k, axis=0)

            start = None
            for i, flag in enumerate(np.append(inside, False)):
                if flag and start is None:
                    start = i
                elif not flag and start is not None:
                    run = pts[start:i]
                    start = None
                    if len(run) < 2:
                        continue
                    arc_length = np.hypot(np.diff(run[:, 0]), np.diff(run[:, 1])).sum()
                    if arc_length >= min_run_length:
                        result.append(np.round(run).astype(np.int32).reshape(-1, 1, 2))
        return result

    def sort_and_limit_contours(self, contours, target_width, target_height, max_paths, faces=None):
        """Sort and limit the number of contours to a specified maximum.
        Contours on/near a face come first; background fills the remaining budget."""
        if faces:
            face_centers = [np.array([(f.left() + f.right()) / 2, (f.top() + f.bottom()) / 2]) for f in faces]

            def priority(c):
                x, y, w, h = cv2.boundingRect(c)
                on_face = any(
                    x < f.right() and x + w > f.left() and y < f.bottom() and y + h > f.top()
                    for f in faces
                )
                centroid = np.array([x + w / 2, y + h / 2])
                dist = min(np.linalg.norm(centroid - fc) for fc in face_centers)
                return (0 if on_face else 1, dist)

            sorted_contours = sorted(contours, key=priority)
        else:
            image_center = np.array([target_width // 2, target_height // 2])
            sorted_contours = sorted(contours, key=lambda c: np.linalg.norm(np.mean(np.squeeze(c, axis=1), axis=0) - image_center))
        return sorted_contours[:max_paths]

    # ----- Facial Feature Filtering -----
    def build_feature_distance_map(self, landmarks_list, width, height):
        """Distance (px) from every pixel to the nearest facial-feature polyline."""
        mask = np.zeros((height, width), np.uint8)
        for landmarks in landmarks_list:
            pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], np.int32)
            for indices in LANDMARK_GROUPS.values():
                cv2.polylines(mask, [pts[indices].reshape(-1, 1, 2)], False, 255, 1)
        return cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)

    def densify_closed(self, pts, max_seg):
        """Insert interpolated points on edges longer than max_seg, treating pts as a closed loop."""
        densified = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            densified.append(a)
            length = np.hypot(*(b - a))
            if length > max_seg:
                steps = int(length // max_seg)
                for t in np.linspace(0, 1, steps + 2)[1:-1]:
                    densified.append(a + t * (b - a))
        return np.array(densified)

    def filter_contours_to_features(self, contours, landmarks_list, width, height, radius=18, gap_tol=2, min_run_length=8):
        """Keep only the sub-segments of each contour that run within `radius` px
        of a facial-feature line. Splits contours; returns (N,1,2) int32 arrays."""
        if not landmarks_list:
            print("No landmarks found: skipping feature filter.")
            return contours

        dist_map = self.build_feature_distance_map(landmarks_list, width, height)
        radius_px = radius * width / 800.0
        filtered = []

        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float64)
            dense = self.densify_closed(pts, radius_px)

            xs = np.clip(dense[:, 0].astype(np.int32), 0, width - 1)
            ys = np.clip(dense[:, 1].astype(np.int32), 0, height - 1)
            inside = dist_map[ys, xs] <= radius_px

            if inside.all():
                # Keep the original contour untouched (no densification artifacts)
                filtered.append(contour)
                continue
            if not inside.any():
                continue

            # Rotate so index 0 is outside: no inside-run straddles the array boundary
            k = int(np.argmin(inside))
            inside = np.roll(inside, -k)
            dense = np.roll(dense, -k, axis=0)

            # Scan maximal inside-runs, bridging short outside gaps to avoid chatter
            runs = []
            start, gap = None, 0
            for i, flag in enumerate(inside):
                if flag:
                    if start is None:
                        start = i
                    gap = 0
                elif start is not None:
                    gap += 1
                    if gap > gap_tol:
                        runs.append((start, i - gap + 1))
                        start, gap = None, 0
            if start is not None:
                runs.append((start, len(inside) - gap))

            for start, end in runs:
                run = dense[start:end]
                if len(run) < 2:
                    continue
                arc_length = np.hypot(np.diff(run[:, 0]), np.diff(run[:, 1])).sum()
                if arc_length < min_run_length:
                    continue
                filtered.append(np.round(run).astype(np.int32).reshape(-1, 1, 2))

        print(f"Feature filter: {len(contours)} contours -> {len(filtered)} feature segments")
        return filtered

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
            random_color = get_random_color()
            dwg.add(dwg.polyline(points, fill="none", stroke=random_color, stroke_width="1"))
            num_paths += 1
        return num_paths

    # ----- SVG Processing -----  
    @profile
    def process_svg(self, svg_filepath, method="none", removal_percentage=80, angle=25):
        """Process the SVG file to align it to points generated by various methods."""
        
        grid_points = None
        
        # Pick a point generation method
        if method == "dynamic_grid":
            print("Using Dynamic Grid method for point generation.")
            grid_points = self.generate_dynamic_points()
        elif method == "poisson_disk":
            print("Using Poisson Disk Sampling method for point generation.")
            grid_points = self.generate_poisson_disk_points()
            
        # SVG Clean up
        if grid_points:
            processed_svg_filepath = svg_filepath.rsplit('.', 1)[0] + '_processed.svg'
            self.align_svg_to_points(svg_filepath, processed_svg_filepath, grid_points)
        else:
            processed_svg_filepath = self.simplify_svg(svg_filepath, removal_percentage=removal_percentage)

        return processed_svg_filepath

    def chain_paths(self, paths, closed_tol=3.0, max_opt_passes=10):
        """Chain polylines into one continuous path, ordering and orienting them so the
        straight connector jumps are as short as possible: greedy nearest-endpoint
        construction refined by 2-opt reversals, path flips and loop re-entry.
        :param paths: List of lists of (x, y) tuples.
        :return: A single list of (x, y) tuples."""
        items = []
        for p in paths:
            if len(p) < 2:
                continue
            pts = np.array(p, dtype=np.float64)
            closed = np.hypot(*(pts[0] - pts[-1])) <= closed_tol
            if closed and np.array_equal(pts[0], pts[-1]):
                pts = pts[:-1]
            items.append({"pts": pts, "closed": closed})
        if not items:
            return []

        # Closed loops enter and exit at pts[0]; open paths run pts[0] -> pts[-1]
        def start(it):
            return it["pts"][0]

        def end(it):
            return it["pts"][0] if it["closed"] else it["pts"][-1]

        def dist(a, b):
            return float(np.hypot(*(a - b)))

        def roll_loop(it, target):
            """Enter a closed loop at the point nearest to target."""
            pts = it["pts"]
            d = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1])
            it["pts"] = np.roll(pts, -int(np.argmin(d)), axis=0)

        def flip(it):
            if not it["closed"]:
                it["pts"] = it["pts"][::-1]
            return it

        # Greedy construction, seeded with the first path: document order is
        # face-priority from sort_and_limit_contours, so the line starts on the face
        seq = [items.pop(0)]
        while items:
            cur = end(seq[-1])
            best_i, best_d = 0, None
            for i, it in enumerate(items):
                pts = it["pts"]
                if it["closed"]:
                    d = float(np.min(np.hypot(pts[:, 0] - cur[0], pts[:, 1] - cur[1])))
                else:
                    d = min(dist(cur, pts[0]), dist(cur, pts[-1]))
                if best_d is None or d < best_d:
                    best_i, best_d = i, d
            it = items.pop(best_i)
            if it["closed"]:
                roll_loop(it, cur)
            elif dist(cur, it["pts"][-1]) < dist(cur, it["pts"][0]):
                it["pts"] = it["pts"][::-1]
            seq.append(it)

        # Refine the order to shorten the total connector length, keeping the
        # first path fixed so the line still starts on the face
        for _ in range(max_opt_passes):
            improved = False

            # Re-enter loops at the point nearest the predecessor's exit
            for k in range(1, len(seq)):
                if seq[k]["closed"]:
                    prev_end = end(seq[k - 1])
                    before = dist(prev_end, start(seq[k]))
                    roll_loop(seq[k], prev_end)
                    if dist(prev_end, start(seq[k])) < before - 1e-9:
                        improved = True

            # Flip open paths when it shortens their two connectors
            for k in range(1, len(seq)):
                if seq[k]["closed"]:
                    continue
                prev_end = end(seq[k - 1])
                nxt = start(seq[k + 1]) if k + 1 < len(seq) else None
                pts = seq[k]["pts"]
                old = dist(prev_end, pts[0]) + (dist(pts[-1], nxt) if nxt is not None else 0)
                new = dist(prev_end, pts[-1]) + (dist(pts[0], nxt) if nxt is not None else 0)
                if new < old - 1e-9:
                    seq[k]["pts"] = pts[::-1]
                    improved = True

            # 2-opt: reverse a sub-sequence when it shortens the two boundary
            # connectors (internal connector lengths are unaffected by reversal)
            for i in range(1, len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    e_prev, s_i, e_j = end(seq[i - 1]), start(seq[i]), end(seq[j])
                    s_next = start(seq[j + 1]) if j + 1 < len(seq) else None
                    old = dist(e_prev, s_i) + (dist(e_j, s_next) if s_next is not None else 0)
                    new = dist(e_prev, e_j) + (dist(s_i, s_next) if s_next is not None else 0)
                    if new < old - 1e-9:
                        seq[i:j + 1] = [flip(it) for it in reversed(seq[i:j + 1])]
                        improved = True

            if not improved:
                break

        chain = []
        for it in seq:
            pts = it["pts"]
            if it["closed"]:
                # Close the loop by repeating the entry point
                pts = np.vstack([pts, pts[:1]])
            chain.extend(map(tuple, pts))
        return chain

    def chain_svg_polylines(self, svg_filepath):
        """Merge all polylines in an SVG into one continuous polyline (one-line drawing)."""
        tree = etree.parse(svg_filepath)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        polylines = root.findall('.//svg:polyline', namespaces=namespace)

        paths = [self.parse_points(p.get('points')) for p in polylines if p.get('points')]
        chained = self.chain_paths(paths)

        for polyline in polylines:
            parent = polyline.getparent()
            if parent is not None:
                parent.remove(polyline)

        if chained:
            etree.SubElement(
                root, '{http://www.w3.org/2000/svg}polyline',
                points=self.points_to_str(chained),
                fill='none', stroke=get_random_color(),
                **{'stroke-width': '1'}
            )

        output_svg_filepath = svg_filepath.rsplit('.', 1)[0] + '_oneline.svg'
        tree.write(output_svg_filepath, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        print(f"Chained {len(paths)} polylines into one line: {output_svg_filepath}")
        return output_svg_filepath

    # Ramer-Douglas-Peucker simplification via OpenCV (C-speed)
    def rdp(self, points, epsilon):
        """
        Simplify a polyline with the Ramer-Douglas-Peucker algorithm.
        :param points: List of (x, y) tuples representing the polyline.
        :param epsilon: Tolerance for simplification.
        :return: List of (x, y) tuples representing the simplified polyline.
        """
        if len(points) < 3:
            return points
        points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(points_np, epsilon, False)
        return [(float(p[0][0]), float(p[0][1])) for p in approx]

    # Helper function to convert points string to list of tuples
    def parse_points(self, points_str):
        return [tuple(map(float, p.split(','))) for p in points_str.split()]

    # Helper function to convert list of tuples back to string
    def points_to_str(self, points):
        return ' '.join(f"{x},{y}" for x, y in points)


    # Angle-based simplification: remove points with small angles between segments
    def angle_simplification(self, points, angle_threshold_deg=15):
        """
        Simplify a polyline by removing points where the angle between two adjacent segments is small.
        :param points: List of (x, y) tuples representing the polyline.
        :param angle_threshold_deg: The angle threshold (in degrees) below which the point will be removed.
        :return: List of simplified (x, y) tuples.
        """
        def angle_between(p1, p2, p3):
            """
            Calculate the angle between two vectors (p1 -> p2) and (p2 -> p3).
            """
            # Vector p1 -> p2
            v1x, v1y = p2[0] - p1[0], p2[1] - p1[1]
            # Vector p2 -> p3
            v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
            # Dot product
            dot_product = v1x * v2x + v1y * v2y
            # Magnitudes
            mag_v1 = math.sqrt(v1x**2 + v1y**2)
            mag_v2 = math.sqrt(v2x**2 + v2y**2)
            # Cosine of the angle
            cos_theta = dot_product / (mag_v1 * mag_v2)
            # Ensure the value is within the valid range for acos
            cos_theta = max(-1, min(1, cos_theta))
            # Return the angle in radians, converted to degrees
            return math.acos(cos_theta) * (180.0 / math.pi)

        simplified = [points[0]]  # Start with the first point
        for i in range(1, len(points) - 1):
            prev_point = simplified[-1]
            current_point = points[i]
            next_point = points[i + 1]
            
            # Calculate the angle between the two segments
            angle = angle_between(prev_point, current_point, next_point)
            
            # If the angle is greater than the threshold, keep the point, otherwise skip it
            if angle > angle_threshold_deg:
                simplified.append(current_point)
        
        simplified.append(points[-1])  # Always keep the last point
        return simplified

    @profile
    def simplify_svg(self, svg_filepath, removal_percentage=80):
        tree = etree.parse(svg_filepath)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        polylines = root.findall('.//svg:polyline', namespaces=namespace)

        print(f"Found {len(polylines)} polylines")
        keep_fraction = 1 - removal_percentage / 100

        # For each polyline, simplify with RDP algorithm
        for i, polyline in enumerate(polylines):
            points = polyline.get('points')
            if points:
                parsed_points = self.parse_points(points)
                epsilon = 1.0 * keep_fraction  # Adjust epsilon based on keep_fraction
                simplified_points = self.rdp(parsed_points, epsilon * 10)
                simplified_points_str = self.points_to_str(simplified_points)
                polyline.set('points', simplified_points_str)

        # Remove unnecessary elements
        xpath_expr = './/*[not(@points) and not(@d) and not(@x) and not(@y)]'
        for elem in root.xpath(xpath_expr, namespaces=namespace):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)

        # Generate output file path for the simplified SVG
        output_svg_filepath = svg_filepath.rsplit('.', 1)[0] + f'_{removal_percentage}_simplified.svg'
        tree.write(output_svg_filepath, pretty_print=True, xml_declaration=True, encoding='UTF-8')

        print(f"Simplified SVG saved to {output_svg_filepath}")

        # Call remove_duplicate_segments to remove any duplicate segments from the SVG
        cleaned_svg_filepath = self.remove_duplicate_segments(output_svg_filepath)

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
    def remove_duplicate_segments(self, svg_filepath, threshold=2.0):
        assert isinstance(svg_filepath, str), "Expected svg_filepath to be a string path"
        
        # Nested helper functions... (same as your original code)
        def _parse_points(points_str):
            # ... (same)
            points = points_str.strip().replace(' ', ',').split(',')
            result = []
            for i in range(0, len(points), 2):
                result.append((float(points[i]), float(points[i+1])))
            return result

        def _format_points(points):
            # ... (same)
            return ' '.join(f"{x},{y}" for x, y in points)

        def _distance(p1, p2):
            # ... (same)
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        try:
            with open(svg_filepath, 'rb') as f:
                tree = etree.parse(f)
        except Exception as e:
            print(f"Error reading the SVG file {svg_filepath}: {e}")
            return None

        root = tree.getroot()
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        polylines = root.findall('.//svg:polyline', namespaces=namespace)

        if not polylines:
            print("No polylines found in the SVG file.")
            return svg_filepath

        all_segments = []
        
        # Phase 1: Collect all segments from all polylines
        for polyline in polylines:
            points_str = polyline.get('points', '')
            if not points_str:
                continue
            
            points = _parse_points(points_str)
            if len(points) < 2:
                continue
                
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i+1]
                all_segments.append({
                    'segment': tuple(sorted([p1, p2])),
                    'parent_polyline': polyline,
                    'index': i,
                    'is_duplicate': False
                })

        # Phase 2: Identify duplicate segments (KD-tree over endpoint pairs instead of O(n²) scan)
        if all_segments:
            coords = np.array(
                [[s['segment'][0][0], s['segment'][0][1], s['segment'][1][0], s['segment'][1][1]] for s in all_segments]
            )
            kdtree = cKDTree(coords)
            kept = np.zeros(len(all_segments), dtype=bool)
            search_radius = threshold * math.sqrt(2)

            for i, segment_info in enumerate(all_segments):
                p1, p2 = segment_info['segment']
                is_dup = False
                for j in kdtree.query_ball_point(coords[i], search_radius):
                    if j >= i or not kept[j]:
                        continue
                    s1, s2 = all_segments[j]['segment']
                    if (_distance(p1, s1) <= threshold and _distance(p2, s2) <= threshold) or \
                       (_distance(p1, s2) <= threshold and _distance(p2, s1) <= threshold):
                        is_dup = True
                        break
                if is_dup:
                    segment_info['is_duplicate'] = True
                else:
                    kept[i] = True

        # Fast lookup for Phase 3: (polyline, segment index) -> segment info
        segment_lookup = {(id(s['parent_polyline']), s['index']): s for s in all_segments}

        # Phase 3: Split and modify polylines
        for polyline in polylines:
            points_str = polyline.get('points', '')
            if not points_str:
                continue
                
            original_points = _parse_points(points_str)
            parent = polyline.getparent()

            if parent is None:
                continue
            
            segments_to_remove_indices = []
            for i in range(len(original_points) - 1):
                segment_info = segment_lookup.get((id(polyline), i))
                if segment_info is not None and segment_info['is_duplicate']:
                    segments_to_remove_indices.append(i)
            
            # If no duplicates found in this polyline, continue to the next one
            if not segments_to_remove_indices:
                continue

            # Split the polyline based on duplicate segments
            start_index = 0
            
            # Add a sentinel value to handle the last segment correctly
            segments_to_remove_indices.append(len(original_points) - 1) 
            
            # Split the path at each duplicate segment
            for i, end_index in enumerate(segments_to_remove_indices):
                
                # The first new path includes the point before the removed segment
                path_points_1 = original_points[start_index : end_index ]

                if len(path_points_1) > 1:
                    # Create a new polyline for this path
                    new_polyline = etree.Element('polyline')
                    new_polyline.set('points', _format_points(path_points_1))
                    
                    # Copy other attributes like style, stroke, etc.
                    for key, value in polyline.attrib.items():
                        if key != 'points':
                            new_polyline.set(key, value)
                    
                    parent.append(new_polyline)

                # The next path will start from the point after the removed segment
                start_index = end_index + 1
            
            # After splitting, remove the original polyline
            parent.remove(polyline)


        # Generate and write the output file
        output_svg_filepath = svg_filepath.rsplit('.', 1)[0] + '_reduced.svg'
        tree.write(output_svg_filepath, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        print(f"SVG saved with duplicates removed to: {output_svg_filepath}")

        return output_svg_filepath



    def get_svgpath_length(self, svg_filepath):
        """
        Calculates the total length of all polylines in an SVG file.
        
        :param svg_filepath: Path to the SVG file
        :return: Total length of all polylines in the SVG
        """
        def distance(p1, p2):
            """Calculate the Euclidean distance between two points."""
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def parse_points(points_str):
            """Parse the 'points' attribute into a list of (x, y) tuples."""
            return [tuple(map(float, point.split(','))) for point in points_str.strip().split()]

        # Parse the SVG file
        tree = etree.parse(svg_filepath)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        polylines = root.findall('.//svg:polyline', namespaces=namespace)

        total_length = 0.0

        for polyline in polylines:
            points = polyline.get('points')
            if points:
                parsed_points = parse_points(points)
                # Calculate length of the polyline
                length = sum(distance(parsed_points[i], parsed_points[i + 1]) for i in range(len(parsed_points) - 1))
                total_length += length

        return int(round(total_length))
    
    # ----- SVG Output -----
    @profile  
    def create_output_svg(self, image_svg_path, imgname='image', scale_factor=0.3, offset_x=0, offset_y=0, id=0, paper_width=500, paper_height=500):
        """Create output image on artboard with id for output position"""
        # Load the original SVG content from a file
        with open(image_svg_path, 'rb') as file:  # Note 'rb' mode for reading as bytes
            svg_data = file.read()

        # Parse the original SVG
        root = etree.fromstring(svg_data)

        # Create a new SVG drawing with svgwrite, setting the desired size and viewBox
        dwg = svgwrite.Drawing(
            size=(paper_width, paper_height),
            profile='full',
            viewBox=f'0 0 {paper_width} {paper_height}'
        )
        dwg.attribs.update({
            "xmlns": "http://www.w3.org/2000/svg"
        })

        # Transform and position the image
        group = dwg.g(id="all_paths", transform=f"translate({offset_x}, {offset_y}) scale({scale_factor})")

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
    def collect_all_paths(self, input_directory, output_file, prefix=""):
        """Combines all SVG files in the input directory. """
        # Get all SVG files in the directory, sorted alphabetically
        svg_files = sorted(
            f for f in os.listdir(input_directory) 
            if f.endswith('.svg') and (f.startswith(prefix) if prefix else True)
        )
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
    
    # ----- Face landmarks -----
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
        blended_image = cv2.addWeighted(image, 0.8, enhanced_image, 0.2, 0)
        return blended_image

    @profile
    def draw_facial_landmarks(self, image, face_rect):
        """
        Draw a random subset of feature lines from the 68 facial landmarks into the image
        and return the detected landmarks.
        """
        if image is None or image.size == 0:
            print("Error: Image is empty or not loaded properly.")
            return None

        landmarks = self.landmark_detector(image, face_rect)

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

        return landmarks

    def draw_feature_line(self, img, landmarks, points_indices, color=(255, 255, 255), thickness=2):
        """Helper method to draw lines connecting facial landmarks."""
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in points_indices if 0 <= i < 68]

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, thickness)
