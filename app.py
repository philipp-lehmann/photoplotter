from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Define directories for your snapped and traced images
SNAPPED_DIR = os.path.join(os.getcwd(), 'photos/snapped')
TRACED_DIR = os.path.join(os.getcwd(), 'photos/traced')

@app.route('/')
def index():
    # Get the list of image files in the snapped folder
    images = sorted([f for f in os.listdir(SNAPPED_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))])
    triplets = []

    # Group images in triplets (original, optimized, traced)
    for i in range(0, len(images), 2):
        original = images[i]
        optimized = images[i + 1] if i + 1 < len(images) else None
        base_name = os.path.splitext(original)[0]
        traced = f"{base_name}.svg"
        triplets.append({
            'original': original,
            'optimized': optimized,
            'traced': traced if os.path.exists(os.path.join(TRACED_DIR, traced)) else None
        })

    return render_template('index.html', triplets=triplets)

@app.route('/photos/snapped/<filename>')
def serve_snapped_image(filename):
    return send_from_directory(SNAPPED_DIR, filename)

@app.route('/photos/traced/<filename>')
def serve_traced_image(filename):
    return send_from_directory(TRACED_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
