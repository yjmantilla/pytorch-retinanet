import os
import sys
import rawpy
from PIL import Image

def open_nef_as_image(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    image = Image.fromarray(rgb)
    return image

def resize_images(input_dir, output_dir, size=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            img_path = os.path.join(root, file)
            if file.lower().endswith('.nef'):
                img = open_nef_as_image(img_path)
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(img_path)
            else:
                continue
            
            if size is not None:
                img = img.resize(size)
                
            # Create output directory structure
            rel_dir = os.path.relpath(root, input_dir)
            out_dir = os.path.join(output_dir, rel_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            # Save image to output directory
            img.save(os.path.join(out_dir, file.rsplit('.', 1)[0] + '.jpg'), 'JPEG')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resize_images.py input_dir output_dir [size]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    size = None
    if len(sys.argv) == 4:
        size = int(sys.argv[3]), int(sys.argv[3]) # assuming square images
    resize_images(input_dir, output_dir, size)
