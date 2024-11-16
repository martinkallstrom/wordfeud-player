import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pytesseract


class ScreenshotReader:
    def __init__(self):
        """Initialize the reader with parameters for board detection."""
        # Adjusted color ranges for bonus squares (in BGR)
        self.blue_range = {
            "lower": np.array(
                [150, 50, 0], dtype=np.uint8
            ),  # More permissive blue range
            "upper": np.array([255, 150, 100], dtype=np.uint8),
        }
        self.burgundy_range = {
            "lower": np.array(
                [0, 0, 100], dtype=np.uint8
            ),  # More permissive burgundy range
            "upper": np.array([100, 100, 255], dtype=np.uint8),
        }
        # Add noise removal parameters
        self.kernel_size = (5, 5)  # Smaller kernel for finer details
        self.erode_iterations = 2  # Fewer erosion iterations
        self.dilate_iterations = 2  # Matching dilate iterations

    @staticmethod
    def get_board_boundaries(
        image: np.ndarray,
        blue_range: Dict[str, np.ndarray],
        burgundy_range: Dict[str, np.ndarray],
        kernel_size: Tuple[int, int] = (5, 5),
        erode_iterations: int = 2,
        dilate_iterations: int = 2
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Find the board boundaries using the white points in the combined bonus square mask.
        Returns ((min_x, min_y), (max_x, max_y))
        
        Args:
            image: OpenCV image in BGR format
            blue_range: Dict with 'lower' and 'upper' BGR bounds for blue squares
            burgundy_range: Dict with 'lower' and 'upper' BGR bounds for burgundy squares
            kernel_size: Size of kernel for morphological operations
            erode_iterations: Number of erosion iterations
            dilate_iterations: Number of dilation iterations
            
        Returns:
            Tuple of ((min_x, min_y), (max_x, max_y)) coordinates
        """
        # Add input validation
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Create masks for both colors
        blue_mask = cv2.inRange(
            image, blue_range["lower"], blue_range["upper"]
        )
        burgundy_mask = cv2.inRange(
            image, burgundy_range["lower"], burgundy_range["upper"]
        )

        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, burgundy_mask)

        # Apply morphological operations to remove noise
        kernel = np.ones(kernel_size, np.uint8)
        # First erode to remove small artifacts
        eroded = cv2.erode(combined_mask, kernel, iterations=erode_iterations)
        # Then dilate to restore the remaining shapes
        filtered_mask = cv2.dilate(eroded, kernel, iterations=dilate_iterations)

        # Find all white points in filtered mask
        points = cv2.findNonZero(filtered_mask)

        if points is None or len(points) < 4:  # Need at least 4 points for a rectangle
            raise ValueError(
                "Insufficient bonus squares detected for board boundary calculation"
            )

        # Get min/max coordinates of all white points
        x_coords = points[:, 0, 0]
        y_coords = points[:, 0, 1]

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        # Make the selection square
        width = max_x - min_x
        height = max_y - min_y
        max_size = max(width, height)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        half_size = max_size // 2
        min_x = center_x - half_size
        min_y = center_y - half_size
        max_x = center_x + half_size
        max_y = center_y + half_size

        return ((min_x, min_y), (max_x, max_y))

    def find_board_boundaries(
        self, image: np.ndarray
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Find the board boundaries using the white points in the combined bonus square mask.
        Returns ((min_x, min_y), (max_x, max_y))
        """
        return self.get_board_boundaries(
            image,
            self.blue_range,
            self.burgundy_range,
            self.kernel_size,
            self.erode_iterations,
            self.dilate_iterations
        )

    def create_grid(
        self, boundaries: Tuple[Tuple[int, int], Tuple[int, int]], size: int = 15
    ) -> List[Dict]:
        """
        Create a grid of cells based on board boundaries.
        """
        (min_x, min_y), (max_x, max_y) = boundaries
        width = max_x - min_x
        height = max_y - min_y

        cell_width = width // size
        cell_height = height // size

        cells = []
        for row in range(size):
            for col in range(size):
                x = min_x + (col * cell_width)
                y = min_y + (row * cell_height)

                cells.append(
                    {
                        "box": {
                            "x": x,
                            "y": y,
                            "width": cell_width,
                            "height": cell_height,
                        },
                        "grid_pos": (row, col),
                    }
                )

        return cells

    def match_templates(self, image: np.ndarray) -> List[Dict]:
        """
        Find all letters in the image using template matching.
        """
        # Add template caching to avoid reloading
        if not hasattr(self, "_cached_templates"):
            self._cached_templates = {}
            template_dir = Path("letter_templates")
            for template_path in template_dir.glob("*.png"):
                letter = template_path.stem.split("_")[0]  # Get just the letter part
                try:
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self._cached_templates[letter] = template
                    else:
                        print(f"Failed to load template {template_path}")
                except Exception as e:
                    print(f"Error loading template {template_path}: {e}")

        templates = self._cached_templates
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        matches = []

        # Find board letters (scale 1.0) in entire image
        print("\nSearching for letters...")
        for letter, template in templates.items():
            try:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.8)

                for y, x in zip(*locations):
                    matches.append(
                        {
                            "letter": letter,
                            "confidence": float(result[y, x]),
                            "box": {
                                "x": x,
                                "y": y,
                                "width": template.shape[1],
                                "height": template.shape[0],
                            },
                            "scale": 1.0,
                        }
                    )
            except cv2.error as e:
                print(f"Error matching template for {letter}: {e}")
                continue

        # Remove overlapping matches
        filtered_matches = []
        while matches:
            best = max(matches, key=lambda m: m["confidence"])
            filtered_matches.append(best)
            matches.remove(best)

            # Remove overlapping matches
            matches = [
                m
                for m in matches
                if not self._boxes_overlap(best["box"], m["box"], 0.5)
            ]

        return filtered_matches

    def _boxes_overlap(
        self, box1: Dict, box2: Dict, overlap_threshold: float = 0.5
    ) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1, y1 = box1["x"], box1["y"]
        x2, y2 = box2["x"], box2["y"]
        w1, h1 = box1["width"], box1["height"]
        w2, h2 = box2["width"], box2["height"]

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)

        return intersection / min_area > overlap_threshold

    def parse_board_to_array(self, grid_cells: List[Dict], board_letters: List[Dict], boundaries: Tuple[Tuple[int, int], Tuple[int, int]]) -> List[str]:
        """Convert detected letters into a 15x15 text array."""
        # Initialize board with spaces
        board = ["               " for _ in range(15)]
        
        (min_x, min_y), (max_x, max_y) = boundaries
        board_width = max_x - min_x
        board_height = max_y - min_y
        cell_width = board_width / 15
        cell_height = board_height / 15

        # Sort letters by position for consistent processing
        sorted_letters = sorted(board_letters, key=lambda m: (
            int((m['box']['y'] + m['box']['height']/2 - min_y) / cell_height),  # row
            int((m['box']['x'] + m['box']['width']/2 - min_x) / cell_width)     # col
        ))

        for match in sorted_letters:
            box = match["box"]
            # Calculate center point of the letter box
            letter_center_x = box["x"] + box["width"] / 2
            letter_center_y = box["y"] + box["height"] / 2

            # Calculate relative position from board origin
            rel_x = letter_center_x - min_x
            rel_y = letter_center_y - min_y

            # Convert to grid coordinates
            grid_x = int(rel_x / cell_width)
            grid_y = int(rel_y / cell_height)

            # Ensure coordinates are within bounds
            if 0 <= grid_y < 15 and 0 <= grid_x < 15:
                # Convert row to list for modification
                row = list(board[grid_y])
                row[grid_x] = match["letter"].lower()
                board[grid_y] = "".join(row)
        
        return board

    def analyze_screenshot(self, image_path: str) -> Dict:
        """Analyze entire screenshot for letters using template matching."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Find board boundaries first
        print("\nFinding board boundaries...")
        boundaries = self.find_board_boundaries(image)
        print(f"Board boundaries found: {boundaries}")

        # Create grid based on boundaries
        grid_cells = self.create_grid(boundaries)

        # Find board letters using template matching
        print("\nFinding board letters...")
        matches = self.match_templates(image)

        return {
            "boundaries": boundaries,
            "grid_cells": grid_cells,
            "board_letters": matches,
            "board_array": self.parse_board_to_array(grid_cells, matches, boundaries)
        }
