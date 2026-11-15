
import random
import shutil
from pathlib import Path
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_dataset" / "PetImages"
DATASET_DIR = BASE_DIR / "dataset"

# Dataset configuration
TRAIN_PER_CLASS = 1000
VAL_PER_CLASS = 500
SEED = 42

random.seed(SEED)


def is_valid_image(image_path):
    """Check whether an image can be opened and verified."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_valid_images(class_name):
    """Collect valid JPEG images for one class."""
    folder = RAW_DIR / class_name
    valid_images = []

    for image_path in folder.glob("*.jpg"):
        if is_valid_image(image_path):
            valid_images.append(image_path)

    return valid_images


def prepare_class(class_name, output_name):
    print(f"\nPreparing {class_name} images...")

    images = collect_valid_images(class_name)
    random.shuffle(images)

    required = TRAIN_PER_CLASS + VAL_PER_CLASS

    if len(images) < required:
        raise RuntimeError(
            f"Not enough valid images for {class_name}: "
            f"found {len(images)}, need {required}"
        )

    train_images = images[:TRAIN_PER_CLASS]
    val_images = images[TRAIN_PER_CLASS:required]

    destinations = [
        (train_images, DATASET_DIR / "train" / output_name),
        (val_images, DATASET_DIR / "validation" / output_name),
    ]

    for image_list, destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)

        for index, source in enumerate(image_list):
            target = destination / f"{output_name}_{index:04d}.jpg"
            shutil.copy2(source, target)

    print(f"Valid images found: {len(images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")


def main():
    prepare_class("Cat", "cats")
    prepare_class("Dog", "dogs")

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()
