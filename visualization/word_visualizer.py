"""Word placement visualization component."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Set, Tuple

from .font_manager import FontManager
from utils.validation import validate_word_info
from utils.image_processing import get_grid_position, get_pixel_position


class WordVisualizer:
    """Visualizes word placements on the game board."""
    
    def __init__(self):
        """Initialize the word visualizer."""
        self.font_manager = FontManager()

    def _get_existing_letters(
        self,
        results: Dict[str, List[Dict]],
        boundaries: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Find existing letters in the board using grid coordinates."""
        existing_letters = set()
        for match in results["board_letters"]:
            box = match["box"]
            grid_x, grid_y = get_grid_position(
                box["x"] + box["width"] / 2,
                box["y"] + box["height"] / 2,
                boundaries
            )
            existing_letters.add((grid_x, grid_y))
        return existing_letters

    def visualize_word_placement(
        self,
        image: np.ndarray,
        word_info: tuple,
        results: Dict[str, List[Dict]],
        boundaries: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> Image.Image:
        """Draw the word placement on the image with larger letters."""
        validate_word_info(word_info)
        x, y, is_horizontal, word, score = word_info

        # Convert CV2 image to PIL
        debug_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(debug_image)
        draw = ImageDraw.Draw(pil_image)

        # Get existing letters
        existing_letters = self._get_existing_letters(results, boundaries)

        # Calculate cell dimensions from boundaries
        (min_x, min_y), (max_x, max_y) = boundaries
        board_width = max_x - min_x
        board_height = max_y - min_y
        cell_width = board_width / 15
        cell_height = board_height / 15

        # Draw each letter of the word
        base_font = self.font_manager.get_font(40)
        for i, letter in enumerate(word):
            # Calculate grid and pixel positions
            curr_grid_x = x + (i if is_horizontal else 0)
            curr_grid_y = y + (0 if is_horizontal else i)

            # Skip if there's already a letter in this position
            if (curr_grid_x, curr_grid_y) in existing_letters:
                continue

            # Get pixel position for this letter
            pixel_x, pixel_y = get_pixel_position(curr_grid_x, curr_grid_y, boundaries)

            # Calculate font size and position
            font_size = int(min(cell_width, cell_height) * 0.8)
            letter_font = self.font_manager.get_font_variant(40, font_size)

            # Center the letter in the cell
            text_width = letter_font.getlength(letter)
            _, _, text_width, text_height = letter_font.getbbox(letter)
            text_x = pixel_x + (cell_width - text_width) / 2
            text_y = pixel_y + (cell_height - text_height) / 2

            # Draw letter
            draw.text(
                (text_x, text_y),
                letter,
                fill=(255, 0, 0),
                font=letter_font
            )

        return pil_image
