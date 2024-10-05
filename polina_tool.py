import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import warnings
import logging
from sklearn.cluster import KMeans

# Suppress known PIL warnings for PNGs with incorrect sRGB profiles
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.PngImagePlugin")

# Setup logging to capture detailed logs in a file
logging.basicConfig(
    filename='embroidery_tool_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default image path
DEFAULT_IMAGE_PATH = "/Users/danm/Documents/Embroided/test.jpg"

# Log the start of the script
logging.info("Starting embroidery tool script...")

# Function to convert RGB values to a hex string
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_image(image_path):
    logging.info(f"Loading image from path: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error loading image from {image_path}. Please check the file path.")
        print("Error loading image. Please check the file path.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def resize_image(image, max_width=500, max_height=500):
    logging.info("Resizing image...")
    height, width, _ = image.shape
    if height > max_height or width > max_width:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        logging.info(f"Image resized to {new_size}")
        return resized_image
    logging.info("No resizing needed for the image.")
    return image

def display_image(image, title="Image"):
    logging.info(f"Displaying image: {title}")
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def preprocess_image(image, max_colors=10):
    """
    Reduces the image colors to the specified max_colors and saves both the processed image
    and a text file with the RGB values of the colors used.

    Parameters:
    - image: Either a PIL Image object or a NumPy array.
    - max_colors: Maximum number of colors for the reduced image.

    Returns:
    - processed_image: The PIL Image object of the processed image.
    """
    
    # Ensure image is a PIL Image object; convert from NumPy array if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    
    # Convert image to RGB format
    image = image.convert("RGB")
    image_array = np.array(image)

    # Reshape image array to a 2D array of pixels for clustering
    pixels = image_array.reshape((-1, 3))

    # Use K-means clustering to reduce the number of colors
    kmeans = KMeans(n_clusters=max_colors, random_state=42).fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)  # These are the dominant colors
    labels = kmeans.labels_

    # Create the new image using the reduced color palette
    processed_pixels = cluster_centers[labels].reshape(image_array.shape)
    processed_image = Image.fromarray(processed_pixels.astype('uint8'))

    # Save the processed image
    processed_image.save("processed_image.png")

    # Save the cluster centers (unique colors) to a text file
    with open("unique_colors.txt", "w") as f:
        for color in cluster_centers:
            f.write(f"{tuple(color)}\n")  # Write each RGB tuple as a line

    print(f"Processed image saved with {len(cluster_centers)} unique colors: {cluster_centers}.")
    return processed_image, cluster_centers
    """
    Reduces the image colors to the specified max_colors and saves both the processed image
    and a text file with the RGB values of the colors used.
    
    Parameters:
    - image: PIL Image object to be processed.
    - max_colors: Maximum number of colors for the reduced image.
    
    Returns:
    - processed_image: The PIL Image object of the processed image.
    """

    # Convert image to RGB and numpy array format
    image = image.convert("RGB")
    image_array = np.array(image)

    # Reshape image array to a 2D array of pixels for clustering
    pixels = image_array.reshape((-1, 3))

    # Use K-means clustering to reduce the number of colors
    kmeans = KMeans(n_clusters=max_colors, random_state=42).fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)  # These are the dominant colors
    labels = kmeans.labels_

    # Create the new image using the reduced color palette
    processed_pixels = cluster_centers[labels].reshape(image_array.shape)
    processed_image = Image.fromarray(processed_pixels.astype('uint8'))

    # Save the processed image
    processed_image.save("processed_image.png")

    # Save the cluster centers (unique colors) to a text file
    with open("unique_colors.txt", "w") as f:
        for color in cluster_centers:
            f.write(f"{tuple(color)}\n")  # Write each RGB tuple as a line

    print(f"Processed image saved with {len(cluster_centers)} unique colors: {cluster_centers}.")
    return processed_image

def save_processed_image(processed_image, original_image_path):
    """
    Save the processed image in the same directory as the original image.

    Parameters:
        processed_image (ndarray): The processed image array.
        original_image_path (str): The path of the original image.

    Returns:
        None
    """
    logging.info("Saving processed image...")
    directory, filename = os.path.split(original_image_path)
    name, ext = os.path.splitext(filename)
    processed_filename = f"{name}_processed{ext}"
    processed_image_path = os.path.join(directory, processed_filename)

    try:
        # Save the processed image using PIL
        processed_image_pil = Image.fromarray(processed_image)
        processed_image_pil.save(processed_image_path)
        logging.info(f"Processed image saved as '{processed_image_path}'")
    except Exception as e:
        logging.error(f"Failed to save processed image: {e}")
        print(f"Error saving processed image: {e}")

def save_color_list(unique_colors, original_image_path):
    logging.info("Saving color list...")
    directory, filename = os.path.split(original_image_path)
    name, _ = os.path.splitext(filename)
    color_list_filename = f"{name}_color_list.txt"
    color_list_path = os.path.join(directory, color_list_filename)

    try:
        with open(color_list_path, 'w') as color_file:
            color_file.write("List of Unique Colors in Processed Image:\n")
            for i, color in enumerate(unique_colors):
                color_file.write(f"Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})\n")
        logging.info(f"Color list saved as '{color_list_path}'")
    except Exception as e:
        logging.error(f"Failed to save color list: {e}")
        print(f"Error saving color list: {e}")

def main():
    logging.info("Entering main function...")

    image_path = input(f"Please enter the path to your image file (default: {DEFAULT_IMAGE_PATH}): ").strip()
    if not image_path:
        image_path = DEFAULT_IMAGE_PATH

    print("Loading image...")
    logging.info("Loading image...")
    image = load_image(image_path)
    if image is None:
        return

    print("Resizing image...")
    image = resize_image(image)

    print("Displaying the original image...")
    display_image(image, title="Original Image")

    print("Preprocessing the image (reducing to fewer colors)...")
    max_colors = int(input("Enter the number of colors to reduce the image to (10, 20, 30, 100): ").strip())
    if max_colors not in [10, 20, 30, 100]:
        print("Invalid number of colors. Please choose 10, 20, or 30.")
        return

    processed_image, unique_colors = preprocess_image(image, max_colors=max_colors)
    if processed_image is None or unique_colors is None:
        return

    print("Displaying the processed image...")
    display_image(processed_image, title="Processed Image")

    print("Saving the processed image and color list...")
    save_processed_image(processed_image, image_path)  # Save the processed image
    save_color_list(unique_colors, image_path)

    print("Processing complete. Check the log file for more details.")

if __name__ == "__main__":
    main()
