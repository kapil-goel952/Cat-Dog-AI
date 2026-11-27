
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

from model import CatDogCNN


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "catdog_resnet18_best.pth"

device = torch.device("cpu")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = CatDogCNN(pretrained=False).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

image_size = checkpoint["image_size"]

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        checkpoint["mean"],
        checkpoint["std"]
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_idx = probabilities.max(dim=1)

    predicted_class = idx_to_class[predicted_idx.item()]

    print("\n===== CAT-DOG AI PREDICTION =====")

    print("Image:", image_path)
    print("Prediction:", predicted_class)

    print(f"Confidence: {confidence.item() * 100:.2f}%")

    print("\nClass probabilities:")

    for idx, probability in enumerate(probabilities[0]):

        print(
            f"{idx_to_class[idx]}: "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print("Error: Image file not found!")
        sys.exit(1)

    predict_image(image_path)
