"""Test module for board parsing functionality."""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ocr.screenshot_reader import ScreenshotReader

def main():
    # Initialize reader
    reader = ScreenshotReader()
    
    # Process first screenshot in screens directory
    screens_dir = os.path.join(project_root, "screens")
    for filename in os.listdir(screens_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(screens_dir, filename)
            print(f"\nProcessing {filename}...")
            
            try:
                # Analyze screenshot
                results = reader.analyze_screenshot(image_path)
                
                # Print board array in compact format
                print("\nDetected Board:")
                print("state = [")
                for row in results['board_array']:
                    print(f"    \"{row}\",")
                print("]")
                
                # Print number of letters found
                print(f"\nFound {len(results['board_letters'])} letters on the board")
                break  # Just process the first image for now
                
            except Exception as e:
                print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
