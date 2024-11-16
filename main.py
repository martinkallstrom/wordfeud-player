"""Main entry point for the Wordfeud Player application."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from ocr.screenshot_reader import ScreenshotReader
from ocr.read_rack import read_rack_from_screenshot
from visualization.match_visualizer import MatchVisualizer
from visualization.board_visualizer import BoardVisualizer
from visualization.word_visualizer import WordVisualizer
from utils.image_processing import load_image
from wordfeudplayer.wordlist import Wordlist
from wordfeudplayer.board import Board


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


def visualize_word_placement(image, word_info, results, boundaries):
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


class WordfeudPlayer:
    """Main class for the Wordfeud Player application."""
    
    def __init__(self):
        """Initialize the Wordfeud Player."""
        self.screenshot_reader = ScreenshotReader()
        self.match_visualizer = MatchVisualizer()
        self.board_visualizer = BoardVisualizer()
        self.word_visualizer = WordVisualizer()
        
        # Initialize wordlist
        self.wordlist = Wordlist()
        self.wordlist_variant = self.wordlist.read_wordlist("ordlista.txt")
        
        # Initialize board with default bonus squares
        self.board = Board()

    def process_screenshot(self, image_path: str) -> Tuple[Dict, cv2.Mat, Image.Image]:
        """Process a screenshot and return the results.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            Tuple containing:
            - Dict with analysis results including board_array, detected letters, and rack letters
            - OpenCV image
            - PIL image
        """
        # Load and validate image
        cv_image, pil_image = load_image(image_path)
        
        # Process the image with OCR
        results = self.screenshot_reader.analyze_screenshot(image_path)
        
        # Add rack letters to results
        try:
            rack_letters = read_rack_from_screenshot(image_path)
            results['rack_letters'] = rack_letters
            print("\nDetected rack letters:", " ".join(rack_letters))
        except Exception as e:
            print(f"Error detecting rack: {e}")
            results['rack_letters'] = None
        
        return results, cv_image, pil_image

    def print_board_state(self, board_array: List[str]) -> None:
        """Print the current board state in a readable format.
        
        Args:
            board_array: List of 15 strings, each containing 15 characters
        """
        print("\nCurrent Board State:")
        print("state = [")
        for row in board_array:
            print(f"    \"{row}\",")
        print("]")

    def find_moves(self, board_array: List[str], rack_letters: List[str]) -> List[Tuple]:
        """Find best possible moves given current board state and rack letters.
        
        Args:
            board_array: List of 15 strings representing the board
            rack_letters: List of available letters
            
        Returns:
            List of tuples (x, y, horizontal, word, score) for best moves
        """
        try:
            # Update board state
            self.board.set_state(board_array)
            
            # Convert rack letters list to string
            rack_str = "".join(rack_letters).lower()
            
            # Calculate all possible words and scores
            words = self.board.calc_all_word_scores(rack_str, self.wordlist, self.wordlist_variant)
            
            # Get top 20 moves by score
            import heapq
            top_moves = heapq.nlargest(20, words, lambda w: w[4])
            
            return top_moves
        except Exception as e:
            print(f"Error finding moves: {e}")
            return []

    def analyze_board_image(self, image_path: str) -> Dict:
        """
        Analyze a Wordfeud board image and return the analysis results.
        
        Args:
            image_path: Path to the board image file
            
        Returns:
            Dict containing:
            - board_array: List of 15 strings representing the board
            - board_letters: List of detected letters and their positions
            - rack_letters: List of letters in the rack
            - boundaries: Tuple of ((min_x, min_y), (max_x, max_y))
            - visualization: PIL Image of the analyzed board
        """
        try:
            # Process the screenshot
            results, cv_image, pil_image = self.process_screenshot(image_path)
            
            # Get boundaries from results
            boundaries = results.get('boundaries', ((100, 100), (700, 700)))
            
            # If we have both board and rack, find best moves
            if "board_array" in results and "rack_letters" in results and results["rack_letters"]:
                moves = self.find_moves(results["board_array"], results["rack_letters"])
                if moves:
                    # Visualize the best move
                    best_move = moves[0]
                    word_image = visualize_word_placement(cv_image, best_move, results, boundaries)
                    results["visualization"] = Image.fromarray(cv2.cvtColor(word_image, cv2.COLOR_BGR2RGB))
                    results["best_moves"] = moves
            else:
                results["visualization"] = pil_image
            
            return results
        except Exception as e:
            raise Exception(f"Error analyzing board image: {e}")

    def visualize_results(
        self,
        results: Dict,
        cv_image: cv2.Mat,
        pil_image: Image.Image,
        boundaries: tuple,
        word_info: tuple = None
    ) -> None:
        """Print analysis results."""
        # Print the board state
        if "board_array" in results:
            self.print_board_state(results["board_array"])
            print(f"\nFound {len(results.get('board_letters', []))} letters on the board")

        # Print rack state
        if "rack_letters" in results and results["rack_letters"] is not None:
            print("\nRack letters:", " ".join(results["rack_letters"]))

            # Find and print best moves if we have both board and rack
            if "board_array" in results:
                print("\nFinding best moves...")
                moves = self.find_moves(results["board_array"], results["rack_letters"])
                if moves:
                    print('\nBest possible moves:')
                    print('Score StartX StartY  Direction Word (capital letter means "use wildcard")')
                    for x, y, horizontal, word, score in moves:
                        print(
                            "%5d %6d %6s %10s %s"
                            % (score, x, y, "Horizontal" if horizontal else "Vertical", word)
                        )
                        
                    # Visualize the best move
                    best_move = moves[0]
                    word_image = visualize_word_placement(cv_image, best_move, results, boundaries)
                    cv2.imshow("Best Move", word_image)
                    print("\nPress any key to continue to next screenshot...")
                    cv2.waitKey(0)  # Wait for keypress
                    cv2.destroyWindow("Best Move")
                else:
                    print("No valid moves found")

        # Print word placement if provided
        if word_info is not None:
            word, score = word_info[3], word_info[4]
            print(f"\nPotential word placement: {word} (Score: {score})")


def main():
    """Main entry point."""
    # Initialize the player
    player = WordfeudPlayer()
    
    # Process screenshots in the screens directory
    screens_dir = Path("screens")
    if not screens_dir.exists():
        print(f"Please create a 'screens' directory and add screenshots")
        return

    # Process screenshots in sequence
    try:
        for file in screens_dir.iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_path = str(file)
                print(f"\nProcessing {Path(image_path).name}...")

                try:
                    # Use the new analyze_board_image method
                    results = player.analyze_board_image(image_path)
                    
                    # Print the board state
                    if "board_array" in results:
                        player.print_board_state(results["board_array"])
                        print(f"\nFound {len(results.get('board_letters', []))} letters on the board")

                    # Print rack state
                    if "rack_letters" in results and results["rack_letters"] is not None:
                        print("\nRack letters:", " ".join(results["rack_letters"]))

                    # Print best moves if available
                    if "best_moves" in results and results["best_moves"]:
                        print('\nBest possible moves:')
                        print('Score StartX StartY  Direction Word (capital letter means "use wildcard")')
                        for x, y, horizontal, word, score in results["best_moves"]:
                            print(
                                "%5d %6d %6s %10s %s"
                                % (score, x, y, "Horizontal" if horizontal else "Vertical", word)
                            )
                    
                        # Show visualization
                        cv2.imshow("Best Move", cv2.cvtColor(np.array(results["visualization"]), cv2.COLOR_RGB2BGR))
                        print("\nPress any key to continue to next screenshot...")
                        cv2.waitKey(0)
                        cv2.destroyWindow("Best Move")

                except Exception as e:
                    print(f"Error processing screenshot: {e}")
                    continue
                except KeyboardInterrupt:
                    print("\nStopping screenshot processing")
                    cv2.destroyAllWindows()
                    return
    except Exception as e:
        print(f"\nError: {e}")
        return
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
