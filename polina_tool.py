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

def preprocess_image(image, max_colors=10):
    """
    Reduce the image to a specific number of colors for simplified pattern generation.

    Parameters:
        image (ndarray): The input image in RGB format.
        max_colors (int): The number of colors to reduce the image to.

    Returns:
        ndarray: The processed image with reduced colors.
        ndarray: The unique RGB values in the processed image.
    """
    logging.info(f"Starting image preprocessing with {max_colors} colors...")
    start_time = time.time()

    # Convert image to LAB color space for better color segmentation
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    reshaped_image = lab_image.reshape((-1, 3))

    logging.info("Running k-means clustering...")
    # Use k-means clustering to reduce the number of colors
    try:
        kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10, max_iter=300, verbose=0)
        labels = kmeans.fit_predict(reshaped_image)
    except Exception as e:
        logging.error(f"K-means clustering failed: {e}")
        print(f"Error in clustering: {e}")
        return None, None

    logging.info("K-means clustering completed successfully.")

    # Reconstruct the image from the cluster labels and cluster centers
    quantized_image = kmeans.cluster_centers_.astype("uint8")[labels]
    quantized_image = quantized_image.reshape(image.shape)

    # Convert back to RGB to ensure proper color mapping
    final_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    # Get unique RGB colors from the cluster centers
    unique_colors = np.unique(kmeans.cluster_centers_.astype("uint8"), axis=0)

    end_time = time.time()
    logging.info(f"Image preprocessing completed in {end_time - start_time:.2f} seconds.")
    return final_image_rgb, unique_colors

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
