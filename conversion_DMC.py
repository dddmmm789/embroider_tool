# conversion_DMC.py


import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math

# Function to create images showing DMC colors and their details
def create_dmc_image(unique_colors, dmc_matches, output_path='results/dmc_colors_output.png', max_rows_per_image=100):
    """
    Create images showing DMC colors and their details, splitting the output into multiple images if necessary.
    
    Parameters:
        unique_colors (list): List of unique RGB colors.
        dmc_matches (list): List of matched DMC colors for each unique RGB color.
        output_path (str): Path to save the output image.
        max_rows_per_image (int): Maximum number of rows (color samples) per image.
    """
    # Image size settings
    sample_width, sample_height = 100, 50
    padding = 10
    rows = len(unique_colors)
    
    # Determine the number of images required based on the max rows per image
    num_images = (rows // max_rows_per_image) + 1

    # Iterate through each set of rows and create an image
    for img_index in range(num_images):
        start_row = img_index * max_rows_per_image
        end_row = min(start_row + max_rows_per_image, rows)

        # Calculate the height for this image
        num_rows = end_row - start_row
        image_height = num_rows * (sample_height + padding) + padding
        image_width = 800  # Fixed width for labels and color samples

        # Create a blank white image
        img = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Draw each color sample along with its details
        y_offset = padding
        for i, (color, dmc_match) in enumerate(zip(unique_colors[start_row:end_row], dmc_matches[start_row:end_row])):
            # Draw the color sample
            draw.rectangle([padding, y_offset, padding + sample_width, y_offset + sample_height], fill=tuple(color))

            # Write the RGB value and DMC info next to the color sample
            text = f"RGB: {color}  |  DMC {dmc_match['dmc_code']} - {dmc_match['dmc_name']}  |  DMC RGB: {dmc_match['rgb']}"
            draw.text((padding + sample_width + 10, y_offset + 15), text, fill="black", font=font)

            y_offset += sample_height + padding

        # Save the image with an index if there are multiple images
        final_output_path = output_path.replace('.png', f'_{img_index + 1}.png')
        if not os.path.exists(os.path.dirname(final_output_path)):
            os.makedirs(os.path.dirname(final_output_path))
        img.save(final_output_path)
        print(f"Output image saved to: {final_output_path}")


def load_dmc_colors(csv_path='DMC_colors_github.csv'):
    """ Load DMC colors from a CSV file and store them in a DataFrame. """
    dmc_colors = pd.read_csv(csv_path)
    return dmc_colors

def euclidean_distance(rgb1, rgb2):
    """ Calculate the Euclidean distance between two RGB tuples. """
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))

def find_closest_dmc(rgb_value, dmc_colors):
    """
    Find the closest DMC color for the given RGB value using Euclidean distance.

    Parameters:
        rgb_value (tuple): The RGB value to match.
        dmc_colors (DataFrame): A DataFrame containing DMC color information.

    Returns:
        dict: The closest DMC color with its code, name, and RGB values.
    """
    closest_dmc = None
    smallest_distance = float('inf')

    for _, row in dmc_colors.iterrows():
        dmc_rgb = (row['Red'], row['Green'], row['Blue'])
        distance = euclidean_distance(rgb_value, dmc_rgb)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_dmc = {
                'dmc_code': row['Floss#'],
                'dmc_name': row['Description'],
                'rgb': dmc_rgb,
                'distance': distance
            }

    return closest_dmc
