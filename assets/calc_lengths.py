import os
import re
from svgpathtools import svg2paths
from lxml import etree # Used for robust SVG/XML parsing

# --- Configuration ---
FILES_DIR = 'printtest'
NORMALIZED_SIZE = 1000  # Target size for normalization (e.g., 1000x1000px)
# ---------------------

def get_svg_size(svg_file_path):
    """
    Tries to get the size (viewBox or width/height) of the SVG,
    assuming a square canvas.
    Returns the size as a float, or None if it can't be determined.
    """
    try:
        tree = etree.parse(svg_file_path)
        root = tree.getroot()
        
        # 1. Check for viewBox attribute
        viewBox = root.attrib.get('viewBox')
        if viewBox:
            # viewBox is usually "min-x min-y width height"
            parts = [float(p) for p in viewBox.split() if p.strip()]
            if len(parts) == 4 and parts[2] == parts[3]:
                return parts[2]  # Return the width/height (which are equal)

        # 2. Check for width and height attributes
        width_str = root.attrib.get('width')
        height_str = root.attrib.get('height')

        # Simple regex to strip units (like 'px')
        def clean_size(size_str):
            if size_str:
                match = re.match(r"(\d+(\.\d+)?)", size_str.strip())
                if match:
                    return float(match.group(1))
            return None

        width = clean_size(width_str)
        height = clean_size(height_str)
        
        if width is not None and height is not None and width == height:
            return width
        
        # Fallback to a default size if nothing is found (e.g., if you know the typical size)
        # return None
        
    except Exception as e:
        print(f"Error parsing size from {os.path.basename(svg_file_path)}: {e}")
        return None
    return None


def calculate_svg_path_lengths(file_path):
    """
    Calculates the total length of all <path> elements in an SVG file.
    Returns: tuple (total_length, size_px)
    """
    total_length = 0.0
    size_px = get_svg_size(file_path)
    
    try:
        # svg2paths returns a list of Path objects and a list of attributes
        paths, _ = svg2paths(file_path)
        
        for path in paths:
            # The .length() method accurately computes the path length
            total_length += path.length()
            
        return total_length, size_px
    
    except Exception as e:
        # Handles cases where the SVG file might be malformed or non-existent
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None, size_px

def main():
    """Main function to iterate over files and output results."""
    
    # Check if the files directory exists
    if not os.path.isdir(FILES_DIR):
        print(f"Error: Directory '{FILES_DIR}' not found. Please create it and place your SVG files inside.")
        return

    svg_files = [
        os.path.join(FILES_DIR, f) 
        for f in os.listdir(FILES_DIR) 
        if f.lower().endswith('.svg')
    ]
    
    if not svg_files:
        print(f"No SVG files found in '{FILES_DIR}'.")
        return

    results = []

    print(f"--- Processing {len(svg_files)} SVG Files ---")
    
    for file_path in svg_files:
        total_length, size_px = calculate_svg_path_lengths(file_path)
        
        if total_length is not None:
            normalized_length = None
            if size_px and size_px > 0:
                # Normalization: Length / Actual_Size * Target_Size
                normalized_length = (total_length / size_px) * NORMALIZED_SIZE
            
            results.append({
                'filename': os.path.basename(file_path),
                'total_length': total_length,
                'svg_size': size_px,
                'normalized_length': normalized_length
            })

    
    ## Output Results
    
    print("\n" + "="*50)
    print("## SVG Path Length Calculation Results")
    print("="*50)

    for result in results:
        print(f"\n📏 File: **{result['filename']}**")
        print(f"   🖼️  SVG Size: {result['svg_size'] if result['svg_size'] else 'N/A'}x{result['svg_size'] if result['svg_size'] else 'N/A'} (based on viewBox/width/height)")
        print(f"   📐 Total Path Length: {result['total_length']:.2f} user units")
        
        if result['normalized_length'] is not None:
            print(f"   ✨ Normalized Length (to {NORMALIZED_SIZE}x{NORMALIZED_SIZE}): {result['normalized_length']:.2f} units")
        else:
            print("   ✨ Normalized Length: Could not calculate (SVG size unknown)")

if __name__ == "__main__":
    main()