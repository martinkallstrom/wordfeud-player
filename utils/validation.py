"""Validation utilities for input data."""

from typing import Dict, Any, Set


def validate_match_structure(match: Dict[str, Any]) -> None:
    """Validate the structure of a match dictionary."""
    required_keys = {"box", "letter", "confidence", "scale"}
    box_keys = {"x", "y", "width", "height"}

    if not required_keys.issubset(match.keys()):
        raise ValueError(f"Invalid match structure. Required keys: {required_keys}")
    
    if not box_keys.issubset(match["box"].keys()):
        raise ValueError(f"Invalid box structure. Required keys: {box_keys}")


def validate_matches(matches: list[Dict[str, Any]]) -> None:
    """Validate a list of matches."""
    for match in matches:
        validate_match_structure(match)


def validate_word_info(word_info: tuple) -> None:
    """Validate word placement information."""
    if len(word_info) != 5:
        raise ValueError("Word info must contain (x, y, is_horizontal, word, score)")
    
    x, y, is_horizontal, word, score = word_info
    
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("Coordinates (x, y) must be integers")
    
    if not isinstance(is_horizontal, bool):
        raise ValueError("is_horizontal must be a boolean")
    
    if not isinstance(word, str) or not word:
        raise ValueError("Word must be a non-empty string")
    
    if not isinstance(score, (int, float)):
        raise ValueError("Score must be a number")
