from pathlib import Path
import cv2
import shutil


def clean_templates():
    """Process templates and keep only the highest confidence version of each letter."""
    template_dir = Path("letter_templates")
    if not template_dir.exists():
        print("No template directory found!")
        return

    # First, analyze what we have
    templates = list(template_dir.glob("*.png"))
    if not templates:
        print("No templates found in directory!")
        return

    print(f"\nFound {len(templates)} templates in {template_dir}")

    # Group templates by letter and find highest confidence for each
    best_templates = {}
    for template_path in templates:
        try:
            # Parse filename (format: LETTER_CONFIDENCE.png)
            letter, confidence = template_path.stem.split("_")
            confidence = float(confidence)

            if letter not in best_templates or confidence > best_templates[letter][1]:
                best_templates[letter] = (
                    template_path.name,
                    confidence,
                )  # Store just the filename
        except ValueError as e:
            print(f"Warning: Skipping {template_path.name} - {e}")
            continue

    if not best_templates:
        print("No valid templates found!")
        return

    # Show what will be kept
    print("\nWill keep these templates:")
    for letter, (filename, conf) in sorted(best_templates.items()):
        print(f"  {letter}: {filename} ({conf:.0f}%)")

    # Ask for confirmation
    response = input("\nProceed with cleaning templates? (yes/no): ")
    if response.lower() != "yes":
        print("Operation cancelled.")
        return

    # Create backup
    backup_dir = Path("letter_templates_backup")
    if backup_dir.exists():
        print(f"\nBackup directory already exists at {backup_dir}")
        response = input("Overwrite existing backup? (yes/no): ")
        if response.lower() != "yes":
            print("Operation cancelled.")
            return
        shutil.rmtree(backup_dir)

    print(f"\nBacking up templates to {backup_dir}")
    shutil.copytree(template_dir, backup_dir)

    # Clear template directory
    print("\nRemoving old templates...")
    for f in templates:
        f.unlink()

    # Save only the best templates from backup
    print("\nSaving best templates:")
    for letter, (filename, confidence) in sorted(best_templates.items()):
        # Source from backup, save to main directory
        backup_path = backup_dir / filename
        new_path = template_dir / f"{letter}.png"
        shutil.copy2(str(backup_path), str(new_path))
        print(f"  Saved {letter}.png (confidence: {confidence:.0f}%)")

    print(f"\nKept {len(best_templates)} templates with highest confidence")
    print(f"Original templates backed up in {backup_dir}")


if __name__ == "__main__":
    clean_templates()
