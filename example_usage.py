from ocr.screenshot_reader import ScreenshotReader
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageFont, ImageDraw
from constants import TILE_WIDTH, TILE_HEIGHT
import os
import numpy as np


def visualize_matches(
    image_path: str,
    matches: List[Dict],
    title: str = "",
):
    """Visualize template matching results using PIL for text rendering."""
    # Improve error handling for image loading
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for text rendering
    pil_image = Image.fromarray(cv_image)
    draw = ImageDraw.Draw(pil_image)

    # Load base font
    base_font = load_font(40)

    # Draw matches
    for match in matches:
        box = match["box"]
        color = (0, 255, 0) if match["scale"] == 1.0 else (255, 0, 0)

        # Draw rectangle
        draw.rectangle(
            [(box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"])],
            outline=color,
            width=2,
        )

        # Draw confidence above box
        conf_text = f"{match['confidence']:.2f}"
        draw.text(
            (box["x"], box["y"] - 20),
            conf_text,
            fill=color,
            font=base_font.font_variant(size=int(box["height"] / 8)),
        )

        # Draw letter centered in the box
        letter = match["letter"]
        font_size = int(box["height"] * 0.8)
        letter_font = base_font.font_variant(size=font_size)

        text_width, text_height = draw.textsize(letter, font=letter_font)
        text_x = box["x"] + (box["width"] - text_width) // 2
        text_y = box["y"] + (box["height"] - text_height) // 2

        draw.text((text_x, text_y), letter, fill=color, font=letter_font)

    # Convert back to matplotlib-compatible format
    plt.figure(figsize=(15, 15))
    plt.imshow(pil_image)
    plt.title(title)
    plt.axis("off")
    plt.show()

    # Add validation for match structure
    required_keys = {"box", "letter", "confidence", "scale"}
    box_keys = {"x", "y", "width", "height"}
    for match in matches:
        if not required_keys.issubset(match):
            raise ValueError(f"Invalid match structure. Required keys: {required_keys}")
        if not box_keys.issubset(match["box"]):
            raise ValueError(f"Invalid box structure. Required keys: {box_keys}")


def load_font(desired_size: int) -> ImageFont.FreeTypeFont:
    """Load a suitable font supporting Swedish characters."""
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS bold
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux bold
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows bold
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows fallback
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, desired_size)
        except OSError:
            continue
    print("Warning: Using default font, Swedish characters might not display correctly")
    return ImageFont.load_default()


def visualize_detections(image, results, boundaries):
    """Draw detected tiles, grid, and coordinates on the image."""
    debug_image = image.copy()
    (min_x, min_y), (max_x, max_y) = boundaries
    board_width = max_x - min_x
    board_height = max_y - min_y

    # Draw board boundaries
    cv2.rectangle(debug_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Draw grid
    cell_width = board_width / 15
    cell_height = board_height / 15

    # Draw vertical lines
    for i in range(16):
        x = int(min_x + i * cell_width)
        cv2.line(debug_image, (x, min_y), (x, max_y), (0, 255, 0), 1)

    # Draw horizontal lines
    for i in range(16):
        y = int(min_y + i * cell_height)
        cv2.line(debug_image, (min_x, y), (max_x, y), (0, 255, 0), 1)

    # Draw detected letters
    for match in results["board_letters"]:
        box = match["box"]
        x, y = box["x"], box["y"]
        w, h = box["width"], box["height"]

        # Draw rectangle around letter
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate grid position
        rel_x = x - min_x
        rel_y = y - min_y
        grid_x = int(rel_x / cell_width)
        grid_y = int(rel_y / cell_height)

        # Draw letter and coordinates
        text = f"{match['letter']} ({grid_x},{grid_y})"
        cv2.putText(
            debug_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
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


def visualize_word_placement(image, word_info, boundaries):
    """Draw the word placement on the image with larger letters, skipping existing letters."""
    debug_image = image.copy()
    x, y, is_horizontal, word, score = word_info
    (min_x, min_y), (max_x, max_y) = boundaries

    # Calculate cell dimensions
    board_width = max_x - min_x
    board_height = max_y - min_y
    cell_width = board_width / 15
    cell_height = board_height / 15

    # Calculate starting position in pixels
    start_x = min_x + (x * cell_width)
    start_y = min_y + (y * cell_height)

    # Find existing letters in the board using grid coordinates
    existing_letters = set()  # Using a set for faster lookups
    for match in results["board_letters"]:
        box = match["box"]
        grid_x = int((box["x"] + box["width"] / 2 - min_x) / cell_width)
        grid_y = int((box["y"] + box["height"] / 2 - min_y) / cell_height)
        existing_letters.add((grid_x, grid_y))

    # Convert to PIL Image for text rendering
    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(debug_image)
    draw = ImageDraw.Draw(pil_image)

    # Load base font
    base_font = load_font(40)  # Base size, will be adjusted per cell

    # Draw each letter of the word
    for i, letter in enumerate(word):
        # Calculate grid position directly
        curr_grid_x = x + (i if is_horizontal else 0)
        curr_grid_y = y + (0 if is_horizontal else i)

        # Skip if there's already a letter in this position
        if (curr_grid_x, curr_grid_y) in existing_letters:
            continue

        # Calculate pixel position for drawing
        curr_x = int(start_x + (i * cell_width if is_horizontal else 0))
        curr_y = int(start_y + (0 if is_horizontal else i * cell_height))

        # Draw filled pale yellow rectangle with 5px padding
        cell_padding = 5
        draw.rectangle(
            [
                (curr_x + cell_padding, curr_y + cell_padding),
                (
                    int(curr_x + cell_width - cell_padding),
                    int(curr_y + cell_height - cell_padding),
                ),
            ],
            fill=(201, 255, 153),  # Kiwi green
            outline=(150, 150, 150),  # Light gray border
        )

        # Calculate font size to fill most of the cell (accounting for padding)
        font_size = int(min(cell_width, cell_height) * 0.7)  # Reduced from 0.8 to 0.7
        letter_font = base_font.font_variant(size=font_size)

        # Get text size for centering using getbbox
        left, top, right, bottom = letter_font.getbbox(letter.upper())
        text_width = right - left
        text_height = bottom - top

        # Adjust vertical positioning by reducing the top offset
        text_x = curr_x + (cell_width - text_width) / 2
        text_y = (
            curr_y + (cell_height - text_height) / 2 - (text_height * 0.1)
        )  # Subtract 10% of text height

        # Draw letter
        draw.text(
            (text_x, text_y),
            letter.upper(),
            fill=(0, 0, 0),
            font=letter_font,
        )

    # Draw score
    draw.text((10, 10), f"Score: {score}", fill=(0, 0, 0), font=base_font)  # Black

    # Convert back to OpenCV format for display
    debug_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return debug_image


# Initialize reader
reader = ScreenshotReader()

# Get the directory where the script is located
script_dir = Path(__file__).parent.resolve()
screens_dir = script_dir / "screens"

# Create screens directory if it doesn't exist
screens_dir.mkdir(exist_ok=True)

# Replace the image file search section with improved pattern matching
image_files = []
for pattern in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
    image_files.extend(screens_dir.glob(pattern))

image_files = sorted(image_files)

if not image_files:
    print(f"No image files found in {screens_dir}")
    print(
        "Please add some PNG or JPEG images to the 'screens' directory and run again."
    )
    exit(0)

for image_path in image_files:
    print(f"\nFound image: {image_path.name}")
    response = input("Process this image? (y/n, or 'q' to quit): ").lower()

    if response == "q":
        print("Quitting...")
        break
    elif response != "y":
        print("Skipping...")
        continue

    print(f"\nProcessing {image_path}...")
    try:
        results = reader.analyze_screenshot(str(image_path))
        print(f"Found {len(results['board_letters'])} board letters")

        # Debug output for all detected letters
        print("\nDetected letters with coordinates:")
        for match in results["board_letters"]:
            box = match["box"]
            print(
                f"Letter: {match['letter']:<3} | Confidence: {match['confidence']:.2f} | "
                f"Position: ({box['x']}, {box['y']}) | "
                f"Size: {box['width']}x{box['height']} | "
                f"Scale: {match['scale']}"
            )

        # Initialize board with spaces instead of dots
        board = [" " * 15 for _ in range(15)]

        # Get image dimensions
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        print(f"\nImage dimensions: {width}x{height}")

        # Find board boundaries using bonus squares
        boundaries = reader.find_board_boundaries(image)
        (min_x, min_y), (max_x, max_y) = boundaries
        board_width = max_x - min_x
        board_height = max_y - min_y

        print(f"Board boundaries: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print(f"Board size: {board_width}x{board_height}")

        cell_width = board_width / 15
        cell_height = board_height / 15
        print(f"Cell size: {cell_width:.1f}x{cell_height:.1f}")
        for match in results["board_letters"]:
            box = match["box"]
            # Calculate center point of the letter box
            letter_center_x = box["x"] + box["width"] / 2
            letter_center_y = box["y"] + box["height"] / 2

            # Calculate relative position from board origin to letter center
            rel_x = letter_center_x - min_x
            rel_y = letter_center_y - min_y

            # Convert to grid coordinates based on center position
            grid_x = int(rel_x / cell_width)
            grid_y = int(rel_y / cell_height)

            # Calculate cell center coordinates
            cell_center_x = min_x + (grid_x * cell_width) + (cell_width / 2)
            cell_center_y = min_y + (grid_y * cell_height) + (cell_height / 2)

            # Calculate distance from letter center to cell center
            dx = letter_center_x - cell_center_x
            dy = letter_center_y - cell_center_y
            distance = ((dx**2) + (dy**2)) ** 0.5

            print(f"\nLetter {match['letter']} conversion:")
            print(f"  Letter center: ({letter_center_x:.1f}, {letter_center_y:.1f})")
            print(f"  Cell center: ({cell_center_x:.1f}, {cell_center_y:.1f})")
            print(f"  Relative pos: ({rel_x:.1f}, {rel_y:.1f})")
            print(f"  Grid pos: ({grid_x}, {grid_y})")
            print(f"  Distance to cell center: {distance:.1f} pixels")

            # Ensure coordinates are within bounds
            if 0 <= grid_y < 15 and 0 <= grid_x < 15:
                row = list(board[grid_y])
                row[grid_x] = match["letter"].lower()
                board[grid_y] = "".join(row)

        # Print the board with spaces for empty squares (not dots)
        print("\nCurrent Board:")
        for row in board:
            print(row)

        # Prompt user for rack tiles
        rack_tiles = input("Enter your rack tiles: ").strip().lower()

        # Import the function from example_player.py
        from example_player import find_best_moves

        # Calculate best moves
        top20 = find_best_moves(board, rack_tiles)

        # Print the top words
        print("\nTop 20 Words to Play:")
        for x, y, direction, word, score in top20:
            print(
                f"Word: {word}, Score: {score}, Position: ({x}, {y}), Direction: {'Horizontal' if direction else 'Vertical'}"
            )

        # Visualize the top scoring word
        if top20:  # If we have any moves
            top_move = top20[0]  # Get the highest scoring move
            word_visualization = visualize_word_placement(image, top_move, boundaries)

            # Display the visualization
            cv2.imshow(
                "Top Move Visualization",
                cv2.cvtColor(word_visualization, cv2.COLOR_BGR2RGB),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # After processing each image, add a separator
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        print("\nMoving to next image...\n")

print("Processing complete!")
