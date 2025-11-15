"""
Download SuperAnimal-TopViewMouse model from HuggingFace
"""

from pathlib import Path
from dlclibrary import download_huggingface_model

def main():
    # Model directory
    model_dir = Path("./models/superanimal_topviewmouse")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SuperAnimal-TopViewMouse model to: {model_dir}")
    print("This may take a few minutes...")

    try:
        download_huggingface_model("superanimal_topviewmouse", model_dir)
        print(f"\n✅ Model downloaded successfully!")
        print(f"   Location: {model_dir.absolute()}")

        # List downloaded files
        print(f"\n📦 Downloaded files:")
        for file in sorted(model_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.relative_to(model_dir)} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
