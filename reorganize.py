"""
Reorganize project files into proper structure.
Run this script from the project root directory.
"""
import os
import shutil
from pathlib import Path


def move_file_safely(source, destination):
    """Move a file if it exists, create destination directory if needed."""
    source_path = Path(source)
    dest_path = Path(destination)
    
    if source_path.exists():
        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        try:
            shutil.move(str(source_path), str(dest_path))
            print(f"✓ Moved: {source_path} → {dest_path}")
        except Exception as e:
            print(f"✗ Error moving {source_path}: {e}")
    else:
        print(f"⊘ Not found: {source_path}")


def reorganize_project():
    """Reorganize the project structure."""
    print("🔄 Starting project reorganization...\n")
    
    # Data files
    print("📊 Moving data files...")
    move_file_safely("expression_labelled.csv", "data/expression_labelled.csv")
    move_file_safely("GSE19804_series_matrix.txt", "data/GSE19804_series_matrix.txt")
    move_file_safely("expression_labelled.zip", "data/expression_labelled.zip")
    
    # Notebooks
    print("\n📓 Moving notebooks...")
    move_file_safely("NoteBooks/01_exploration_GSE19804.ipynb", 
                     "notebooks/01_exploration_GSE19804.ipynb")
    
    # Scripts
    print("\n📜 Moving scripts...")
    move_file_safely("NoteBooks/ajouter_label_binaire.py", 
                     "scripts/ajouter_label_binaire.py")
    
    # Remove old NoteBooks directory if empty
    notebooks_dir = Path("NoteBooks")
    if notebooks_dir.exists():
        try:
            if not any(notebooks_dir.iterdir()):
                notebooks_dir.rmdir()
                print(f"\n✓ Removed empty directory: {notebooks_dir}")
            else:
                print(f"\n⚠ Directory not empty, keeping: {notebooks_dir}")
        except Exception as e:
            print(f"\n⚠ Could not remove {notebooks_dir}: {e}")
    
    print("\n✅ Reorganization complete!")
    print("\n📋 Next steps:")
    print("   1. Review the changes")
    print("   2. Update file paths in notebooks and scripts")
    print("   3. Test that everything works")
    print("   4. Commit the changes to Git")


if __name__ == "__main__":
    # Get confirmation
    print("This script will reorganize your project structure.")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response == 'y':
        reorganize_project()
    else:
        print("Cancelled.")
