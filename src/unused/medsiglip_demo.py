print("Starting")
import os

import numpy as np
import torch
from PIL import Image
from tensorflow.image import resize as tf_resize
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.getenv("HF_TOKEN")
model = AutoModel.from_pretrained("google/medsiglip-448", token=HF_TOKEN).to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448", token=HF_TOKEN)


# Download sample image
# os.system("wget -nc -q https://storage.googleapis.com/dx-scin-public-data/dataset/images/3445096909671059178.png")
# os.system("wget -nc -q https://storage.googleapis.com/dx-scin-public-data/dataset/images/-5669089898008966381.png")
# imgs = [Image.open("3445096909671059178.png").convert("RGB"), Image.open("-5669089898008966381.png").convert("RGB")]


# If you want to reproduce the results from MedSigLIP evals, we recommend a
# resizing operation with `tf.image.resize` to match the implementation with the
# Big Vision library (https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/pp/ops_image.py#L84).
# Otherwise, you can rely on the Transformers image processor's built-in
# resizing (done automatically by default and uses `PIL.Image.resize`) or use
# another resizing method.
def resize(image):
	return Image.fromarray(
		tf_resize(
			images=image, size=[448, 448], method='bilinear', antialias=False
		).numpy().astype(np.uint8)
	)


# resized_imgs = [resize(img) for img in imgs]

# texts = [
# 	"a photo of an arm with no rash",
# 	"a photo of an arm with a rash",
# 	"a photo of a leg with no rash",
# 	"a photo of a leg with a rash"
# ]

def get_matches(texts, imgs, device="cuda"):
	imgs = [Image.fromarray(img) for img in imgs]
	inputs = processor(text=texts, images=imgs, padding="max_length", return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	logits_per_image = outputs.logits_per_image
	probs = torch.softmax(logits_per_image, dim=1)

	# for n_img, img in enumerate(imgs):
	# # display(img)  # Note this is an IPython function that will only work in a Jupyter notebook environment
	# for i, label in enumerate(texts):
	# 	print(f"{probs[n_img][i]:.2%} that image is '{label}'")

	return outputs

# Get the image and text embeddings
# print(f"image embeddings: {outputs.image_embeds}")
# print(f"text embeddings: {outputs.text_embeds}")
