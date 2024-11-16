"""Image processing utilities."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional


def load_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """Load an image and return both CV2 and PIL versions."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    
    return cv_image, pil_image


def calculate_grid_dimensions(boundaries: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[float, float, float, float]:
    """Calculate grid dimensions from boundaries."""
    (min_x, min_y), (max_x, max_y) = boundaries
    board_width = max_x - min_x
    board_height = max_y - min_y
    
    cell_width = board_width / 15
    cell_height = board_height / 15
    
    return board_width, board_height, cell_width, cell_height


def get_grid_position(x: int, y: int, boundaries: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
    """Convert pixel coordinates to grid positions."""
    (min_x, min_y), (_, _) = boundaries
    _, _, cell_width, cell_height = calculate_grid_dimensions(boundaries)
    
    rel_x = x - min_x
    rel_y = y - min_y
    grid_x = int(rel_x / cell_width)
    grid_y = int(rel_y / cell_height)
    
    return grid_x, grid_y


def get_pixel_position(grid_x: int, grid_y: int, boundaries: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
    """Convert grid positions to pixel coordinates."""
    (min_x, min_y), (_, _) = boundaries
    _, _, cell_width, cell_height = calculate_grid_dimensions(boundaries)
    
    pixel_x = min_x + (grid_x * cell_width)
    pixel_y = min_y + (grid_y * cell_height)
    
    return int(pixel_x), int(pixel_y)
