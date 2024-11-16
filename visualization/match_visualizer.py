"""Match visualization component."""

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

from .font_manager import FontManager
from utils.validation import validate_matches


class MatchVisualizer:
    """Visualizes template matching results."""
    
    def __init__(self):
        """Initialize the match visualizer."""
        self.font_manager = FontManager()

    def visualize_matches(
        self,
        pil_image: Image.Image,
        matches: List[Dict],
        title: str = "",
    ) -> None:
        """Visualize template matching results using PIL for text rendering."""
        validate_matches(matches)
        
        draw = ImageDraw.Draw(pil_image)
        base_font = self.font_manager.get_font(40)

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
                font=self.font_manager.get_font_variant(40, int(box["height"] / 8)),
            )

            # Draw letter centered in the box
            letter = match["letter"]
            font_size = int(box["height"] * 0.8)
            letter_font = self.font_manager.get_font_variant(40, font_size)

            text_width = letter_font.getlength(letter)
            _, _, text_width, text_height = letter_font.getbbox(letter)
            text_x = box["x"] + (box["width"] - text_width) // 2
            text_y = box["y"] + (box["height"] - text_height) // 2

            draw.text((text_x, text_y), letter, fill=color, font=letter_font)

        # Display with matplotlib
        plt.figure(figsize=(15, 15))
        plt.imshow(pil_image)
        plt.title(title)
        plt.axis("off")
        plt.show()
