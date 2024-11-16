"""Setup file for Wordfeud Player."""
from setuptools import setup, find_packages

setup(
    name="wordfeud-player",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "pillow==11.0.0",
        "pytesseract>=0.3.10",
        "matplotlib==3.8.2",
        "packaging==24.2",
    ],
    python_requires=">=3.10",
)
