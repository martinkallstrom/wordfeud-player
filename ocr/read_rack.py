import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pytesseract
import shutil
from .find_rack import RackFinder

def preprocess_tile(tile: np.ndarray, target_height: int) -> np.ndarray:
    """
    Preprocess a tile image for OCR.
    
    Args:
        tile: Input tile image
        target_height: Height to resize to while maintaining aspect ratio
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(tile.shape) == 3:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold at 50%
    _, binary = cv2.threshold(tile, 127, 255, cv2.THRESH_BINARY)
    
    # Resize to target height maintaining aspect ratio
    aspect = binary.shape[1] / binary.shape[0]
    target_width = int(target_height * aspect)
    resized = cv2.resize(binary, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Apply 2x2 erosion to fatten black letters
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(resized, kernel, iterations=1)
    
    return eroded

def analyze_tile_ratio(tile: np.ndarray) -> str:
    """
    Analyze black/white ratio of a tile to determine if it's empty, wildcard, or likely an O.
    
    Args:
        tile: Binary (black and white) image
        
    Returns:
        '_' for empty (>95% black)
        '*' for wildcard (>95% white)
        'O' otherwise
    """
    # Count black and white pixels
    total_pixels = tile.size
    white_pixels = np.count_nonzero(tile)
    black_pixels = total_pixels - white_pixels
    
    # Calculate ratios
    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    
    # Evaluate ratios
    if black_ratio > 0.95:
        return '_'  # Empty tile
    elif white_ratio > 0.95:
        return '*'  # Wildcard
    else:
        return 'O'  # Likely an O

def recognize_tile(tile: np.ndarray) -> str:
    """
    Perform OCR on a single preprocessed tile.
    
    Args:
        tile: Preprocessed tile image
        
    Returns:
        Recognized character or fallback based on black/white ratio
    """
    # Try OCR first
    config = '--psm 10 -l swe --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ'
    try:
        data = pytesseract.image_to_data(tile, config=config, output_type=pytesseract.Output.DICT)
        # Get highest confidence result above 5%
        confidences = [float(x) for x in data['conf'] if x != '-1']
        texts = [t for i, t in enumerate(data['text']) if data['conf'][i] != '-1']
        
        if confidences and texts:
            max_conf_idx = np.argmax(confidences)
            if confidences[max_conf_idx] >= 5.0:  # 5% confidence threshold
                return texts[max_conf_idx]
    except Exception as e:
        print(f"OCR error: {e}")
    
    # If OCR fails, analyze black/white ratio
    # Threshold the image
    _, binary = cv2.threshold(tile, 127, 255, cv2.THRESH_BINARY)
    return analyze_tile_ratio(binary)

def segment_tiles(
    rack_image: np.ndarray, 
    top_margin: float = 0.05,     # 5% from top
    bottom_margin: float = 0.15,  # 15% from bottom
    left_margin: float = 0.15,    # 15% from left
    right_margin: float = 0.30,   # 30% from right
) -> List[np.ndarray]:  # Returns original tiles only
    """
    Segment a rack image into 7 individual tile images.
    
    Args:
        rack_image: OpenCV image of the rack
        top_margin: Percentage of tile size to crop from top (0.05 = 5%)
        bottom_margin: Percentage of tile size to crop from bottom (0.15 = 15%)
        left_margin: Percentage of tile size to crop from left (0.15 = 15%)
        right_margin: Percentage of tile size to crop from right (0.30 = 30%)
        
    Returns:
        List of original tile images
    """
    height, width = rack_image.shape[:2]
    tile_size = width // 7
    
    # Calculate crop sizes for each edge
    top_crop = int(tile_size * top_margin)
    bottom_crop = int(tile_size * bottom_margin)
    left_crop = int(tile_size * left_margin)
    right_crop = int(tile_size * right_margin)
    
    tiles = []
    for i in range(7):
        # Calculate tile boundaries
        x_start = i * tile_size
        x_end = (i + 1) * tile_size
        
        # Apply margins
        x_start += left_crop
        x_end -= right_crop
        y_start = top_crop
        y_end = height - bottom_crop
        
        # Extract tile
        original_tile = rack_image[y_start:y_end, x_start:x_end]
        tiles.append(original_tile)
    
    return tiles

def read_rack_from_screenshot(screenshot_path: str) -> List[str]:
    """
    Process a single screenshot and return the detected rack letters.
    
    Args:
        screenshot_path: Path to the screenshot image
        
    Returns:
        List of 7 letters/characters, where:
        - Letters A-Z, Å, Ä, Ö for detected letters
        - '_' for empty tiles
        - '*' for wildcards
        - 'O' for unclear tiles
    """
    # Initialize finder
    finder = RackFinder()
    
    # Read image
    img = cv2.imread(screenshot_path)
    if img is None:
        raise ValueError(f"Could not read image: {screenshot_path}")
    
    # Find rack
    rack_loc = finder.find_rack_location(img)
    if rack_loc is None:
        raise ValueError(f"No rack found in: {screenshot_path}")
    
    # Crop the rack area
    rack = img[
        rack_loc["y"]:rack_loc["y"] + rack_loc["height"],
        rack_loc["x"]:rack_loc["x"] + rack_loc["width"]
    ]
    
    # Get tiles and process each one
    tiles = segment_tiles(rack)
    target_height = 40
    recognized = []
    
    for tile in tiles:
        # Preprocess and recognize
        proc_tile = preprocess_tile(tile, target_height)
        char = recognize_tile(proc_tile)
        recognized.append(char)
    
    return recognized

def main():
    # Initialize finder
    finder = RackFinder()
    
    # Clear and recreate debug_output directory
    debug_dir = Path('debug_output')
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(exist_ok=True)
    
    # Process all screenshots
    screens_dir = Path('screens')
    for screen_path in screens_dir.glob('*.[pPjJ][nNpP][gGeE]*'):
        print(f"\nProcessing {screen_path.name}...")
        
        try:
            # Process screenshot
            letters = read_rack_from_screenshot(str(screen_path))
            print("Detected letters:", " ".join(letters))
            
            # Save cropped rack
            img = cv2.imread(str(screen_path))
            rack_loc = finder.find_rack_location(img)  # Use same finder instance
            rack = img[
                rack_loc["y"]:rack_loc["y"] + rack_loc["height"],
                rack_loc["x"]:rack_loc["x"] + rack_loc["width"]
            ]
            output_path = debug_dir / f"rack_{screen_path.name}"
            cv2.imwrite(str(output_path), rack)
            
            # Get tiles for visualization
            tiles = segment_tiles(rack)
            target_height = 40
            
            # Plot results
            fig, axes = plt.subplots(2, 7, figsize=(15, 6))
            fig.suptitle(f'OCR Results for {screen_path.name}')
            
            # Plot original tiles in first row
            for i, tile in enumerate(tiles):
                if len(tile.shape) == 3:
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(tile)
                axes[0, i].axis('off')
                axes[0, i].set_title(f'Original {i+1}')
            
            # Plot preprocessed tiles and results
            for col in range(7):
                proc = preprocess_tile(tiles[col], target_height)
                axes[1, col].imshow(proc, cmap='gray')
                axes[1, col].axis('off')
                char = letters[col]
                axes[1, col].set_title(f'{target_height}px\n"{char}"' if char else f'{target_height}px\n(no match)')
            
            plt.tight_layout()
            plt.show()
            
        except ValueError as e:
            print(f"Error: {e}")
            continue

if __name__ == '__main__':
    main()
