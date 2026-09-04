import kagglehub
import shutil
from pathlib import Path

def main():
    # Project's data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # Download dataset from Kaggle
    kaggle_path = kagglehub.dataset_download(
        "nikhilroxtomar/brain-tumor-segmentation"
    )

    # Create project directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy downloaded dataset into project directory
    shutil.copytree(kaggle_path, data_dir, dirs_exist_ok=True)

    print(f"Dataset copied to: {data_dir}")

if __name__ == "__main__":
    main()

