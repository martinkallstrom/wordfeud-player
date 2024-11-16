import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from .screenshot_reader import ScreenshotReader

class RackFinder:
    def __init__(self):
        """Initialize the RackFinder with default parameters."""
        self.side_margin = 10  # pixels from screen edge
        self.empty_rack_threshold = 50  # pixels from bottom
        self.screenshot_reader = ScreenshotReader()
        
    def find_rack_location(self, image: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Find the rack location in a Wordfeud screenshot.
        
        Args:
            image: numpy array of screenshot in BGR format
            
        Returns:
            Dictionary with x, y, width, height of rack location, or None if rack is empty
            
        The algorithm:
        1. Find board boundaries using bonus squares
        2. Calculate rack width based on screen width minus margins
        3. Calculate tile size based on rack width divided by 7
        4. Scan area below board for highest mean intensity rectangle
        5. Returns None if rack bottom edge is too close to screen bottom (empty rack)
        """
        # Convert to grayscale for intensity calculations
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        height, width = gray.shape
        
        # Calculate rack dimensions
        rack_width = width - (2 * self.side_margin)
        tile_size = rack_width // 7
        
        # Try to find board boundaries
        try:
            ((min_x, min_y), (max_x, max_y)) = self.screenshot_reader.find_board_boundaries(image)
            search_start = max_y + 10  # Start 10px below board
        except ValueError:
            # Fall back to bottom third if board boundaries can't be found
            search_start = height * 2 // 3
        
        # Track best position
        best_intensity = -1
        best_y = None
        
        # Scan vertically
        for y in range(search_start, height - tile_size):
            # Define rectangle
            roi = gray[y:y + tile_size, self.side_margin:self.side_margin + rack_width]
            mean_intensity = np.mean(roi)
            
            if mean_intensity > best_intensity:
                best_intensity = mean_intensity
                best_y = y
        
        # Check if rack is empty (too close to bottom)
        if best_y is None or (best_y + tile_size) > (height - self.empty_rack_threshold):
            return None
            
        return {
            "x": self.side_margin,
            "y": best_y,
            "width": rack_width,
            "height": tile_size
        }
        
    def debug_save_rack_crop(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save a cropped image of the detected rack area for debugging.
        
        Args:
            image: numpy array of screenshot in BGR format
            output_path: Path where to save the cropped image
        """
        rack_loc = self.find_rack_location(image)
        if rack_loc is None:
            # For empty racks, save a black rectangle of expected size
            height, width = image.shape[:2]
            rack_width = width - (2 * self.side_margin)
            tile_size = rack_width // 7
            crop = np.zeros((tile_size, rack_width, 3), dtype=np.uint8)
        else:
            # Crop the rack area
            crop = image[
                rack_loc["y"]:rack_loc["y"] + rack_loc["height"],
                rack_loc["x"]:rack_loc["x"] + rack_loc["width"]
            ]
            
        # Save the crop
        cv2.imwrite(str(output_path), crop)
