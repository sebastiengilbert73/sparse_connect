import os
import argparse
from PIL import Image
import re

def create_gif(image_folder, output_path, target_node, duration=500):
    images = []
    # Find all heatmaps for the specific target node
    pattern = re.compile(f"heatmap_epoch_(\d+)_target_{target_node}\.png")
    
    files = [f for f in os.listdir(image_folder) if pattern.match(f)]
    # Sort by epoch number
    files.sort(key=lambda x: int(pattern.match(x).group(1)))
    
    for filename in files:
        img_path = os.path.join(image_folder, filename)
        images.append(Image.open(img_path))
    
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_path}")
    else:
        print(f"No images found for target {target_node} in {image_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF of network evolution.")
    parser.add_argument("--image_folder", default="output_train_network/visualizations", help="Path to heatmaps")
    parser.add_argument("--output_path", default="network_evolution.gif", help="Output GIF path")
    parser.add_argument("--target_node", type=int, required=True, help="Target node ID to visualize")
    parser.add_argument("--duration", type=int, default=500, help="Duration of each frame in ms")
    
    args = parser.parse_args()
    create_gif(args.image_folder, args.output_path, args.target_node, args.duration)
