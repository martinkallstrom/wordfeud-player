"""Board visualization component."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple

from utils.image_processing import get_grid_position


class BoardVisualizer:
    """Visualizes the game board and detections."""
    
    def __init__(self):
        """Initialize the board visualizer."""
        pass

    def draw_grid(
        self,
        image: Image.Image,
        boundaries: Tuple[Tuple[int, int], Tuple[int, int]],
        grid_cells: List[Dict] = None,
    ) -> Image.Image:
        """Draw grid on image based on boundaries."""
        # Create a copy of the image for drawing
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Extract boundaries
        (min_x, min_y), (max_x, max_y) = boundaries
        width = max_x - min_x
        height = max_y - min_y

        # Calculate cell size
        cell_width = width // 15
        cell_height = height // 15

        # Draw vertical lines
        for i in range(16):
            x = min_x + i * cell_width
            draw.line([(x, min_y), (x, max_y)], fill="white", width=2)

        # Draw horizontal lines
        for i in range(16):
            y = min_y + i * cell_height
            draw.line([(min_x, y), (max_x, y)], fill="white", width=2)

        # If grid cells are provided, draw them
        if grid_cells:
            for cell in grid_cells:
                x, y = cell["x"], cell["y"]
                w, h = cell["width"], cell["height"]
                draw.rectangle([(x, y), (x + w, y + h)], outline="yellow", width=1)

        return draw_image

    def visualize_detections(
        self,
        image: np.ndarray,
        results: Dict[str, List[Dict]],
        boundaries: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> np.ndarray:
        """Draw detected tiles and coordinates on the image."""
        pil_image = Image.fromarray(image)
        debug_image = self.draw_grid(pil_image, boundaries)
        debug_image = np.array(debug_image)

        # Draw detected letters
        for match in results["board_letters"]:
            box = match["box"]
            x, y = box["x"], box["y"]
            w, h = box["width"], box["height"]

            # Draw rectangle around letter
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Calculate and draw grid position
            grid_x, grid_y = get_grid_position(x + w/2, y + h/2, boundaries)
            text = f"{match['letter']} ({grid_x},{grid_y})"
            cv2.putText(
                debug_image,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            # Draw confidence score
            conf_text = f"{match['confidence']:.2f}"
            cv2.putText(
                debug_image,
                conf_text,
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return debug_image
