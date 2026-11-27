
import torch
from pathlib import Path
from src.model import CatDogCNN
from PIL import Image
from torchvision import transforms

device = "cpu"

checkpoint = torch.load(
    "models/catdog_best.pth",
    map_location=device,
    weights_only=True
)

print("Class mapping:", checkpoint["class_to_idx"])

model = CatDogCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((checkpoint["image_size"],
                       checkpoint["image_size"])),
    transforms.ToTensor()
])

for path in Path("test_images").iterdir():

    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)

    print("\nImage:", path.name)
    print("Raw scores:", logits.tolist())
    print("Probabilities:", probabilities.tolist())
