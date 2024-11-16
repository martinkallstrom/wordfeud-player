"""Font management utilities for visualization components."""

from PIL import ImageFont
from typing import Optional


class FontManager:
    """Manages font loading and caching for visualization."""
    
    _font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS bold
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux bold
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows bold
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows fallback
    ]

    def __init__(self):
        """Initialize the font manager."""
        self._cache = {}

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a font of the specified size, using cache if available."""
        if size in self._cache:
            return self._cache[size]

        font = self._load_font(size)
        self._cache[size] = font
        return font

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a suitable font supporting Swedish characters."""
        for path in self._font_paths:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        print("Warning: Using default font, Swedish characters might not display correctly")
        return ImageFont.load_default()

    def get_font_variant(self, base_size: int, variant_size: int) -> ImageFont.FreeTypeFont:
        """Get a variant of the base font with a different size."""
        return self.get_font(variant_size)
