import os
from PIL import Image
from conversion_DMC import load_dmc_colors, find_closest_dmc, create_dmc_image
import subprocess

# Set paths for files
input_image_path = 'test.jpg'  # The initial image filename to process
dmc_csv_path = 'DMC_colors_github.csv'  # Path to the DMC color CSV file
processed_image_path = '/Users/danm/Documents/Embroided/test_processed.jpg'  # Path where polina_tool.py saves the processed image
output_image_path = 'results/dmc_colors_output.jpg'  # Path for the final output image with DMC colors and details

# Ensure necessary directories exist
os.makedirs('results', exist_ok=True)

# Step 1: Run `polina_tool.py` as a subprocess to handle initial processing
print("Running initial processing with polina_tool.py...")
try:
    subprocess.run(['python', 'polina_tool.py'], check=True)
    print("polina_tool.py executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running polina_tool.py: {e}")
    exit(1)

# Step 2: Verify if the processed image exists
if not os.path.exists(processed_image_path):
    print(f"Processed image not found at {processed_image_path}. Exiting...")
    exit(1)

# Step 3: Load the processed image and extract unique RGB colors
print("Loading the processed image and extracting unique colors...")
processed_image = Image.open(processed_image_path)
processed_image = processed_image.convert("RGB")  # Ensure it's in RGB format
unique_colors = processed_image.getcolors(processed_image.size[0] * processed_image.size[1])

# Convert the list of (count, color) tuples into a list of RGB colors
unique_rgb_colors = [color for count, color in unique_colors]

# Step 4: Load DMC colors from the CSV
print("Loading DMC colors...")
dmc_colors = load_dmc_colors(dmc_csv_path)

# Step 5: Find the closest DMC match for each unique color
print("Finding closest DMC matches...")
dmc_matches = [find_closest_dmc(color, dmc_colors) for color in unique_rgb_colors]

# Step 6: Create the final output image with DMC details
print("Creating output image with DMC colors and details...")
create_dmc_image(unique_rgb_colors, dmc_matches, output_path=output_image_path)

print("Process completed successfully!")
