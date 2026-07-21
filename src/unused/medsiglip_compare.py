print("start")

import util

with util.Timer():
	from training.xray import xray_data
	from models import xray_cnn
	import torchvision
	import numpy as np
	# import llm_test
	import medsiglip_demo
	import torch
	from torchvision.transforms import v2
	import matplotlib.pyplot as plt
	import sys

device = "cuda:0"

resolution = 448
transform = v2.Compose([
	v2.Resize(size=None, max_size=resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
	v2.CenterCrop([resolution, resolution]),
	v2.ToDtype(torch.uint8),
	v2.ToImage()
])
dataset = xray_data.XrayDataset(transform=transform, use_chunks=False)

samples = [dataset[i] for i in range(1000, 1000 + 4 * 64, 4)]
imgs = [sample[0][0] for sample in samples]
views = [sample[0][1] for sample in samples]
targets = [sample[1] for sample in samples]

import visualize

fig, ax = visualize.vis_batch(imgs, nrow=8)
fig.savefig("siglip_batch.jpg", dpi=1000)
# sys.exit()

labels1 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
labels2 = [f"Chest X-ray with {lbl}" for lbl in labels1]
labels3 = [f"An image of a chest X-ray showing {lbl}" for lbl in
           ["atelectasis", "cardiomegaly", "a pulmonary consolidation", "an edema", "an enlarged cardiomediastinum", "a fracture", "a lung lesion", "a lung opacity", "no pathological finding", "a pleural effusion", "pleural disease",
            "pneumonia", "a pneumothorax", "support devices"]]

np_imgs = [img[0].numpy().astype(np.uint8) for img in imgs]
siglip_matches1 = medsiglip.get_matches(imgs=np_imgs, texts=labels1, device=device)
siglip_matches2 = medsiglip.get_matches(imgs=np_imgs, texts=labels2, device=device)
siglip_matches3 = medsiglip.get_matches(imgs=np_imgs, texts=labels3, device=device)
logits1 = siglip_matches1.logits_per_image
logits2 = siglip_matches2.logits_per_image
logits3 = siglip_matches3.logits_per_image
# logits_per_image = siglip_matches.logits_per_image
# probs = torch.softmax(logits_per_image, dim=1)
# probs = torch.softmax(logits_per_image, dim=1)
# probs = logits_per_image

fig, axs = plt.subplots(1, 4)

m1 = torch.softmax(logits1, dim=1).cpu().numpy()
m2 = torch.softmax(logits2, dim=1).cpu().numpy()
m3 = torch.softmax(logits3, dim=1).cpu().numpy()
m4 = targets

axs[0].imshow(m1, aspect="equal")
axs[0].set_title("logits1")
# axs[0].set_aspect("equal")
axs[0].axis("off")

axs[1].imshow(m2, aspect="equal")
axs[1].set_title("logits2")
# axs[1].set_aspect("equal")
axs[1].axis("off")

axs[2].imshow(m3, aspect="equal")
axs[2].set_title("logits3")
# axs[2].set_aspect("equal")
axs[2].axis("off")

axs[3].imshow(m4, aspect="equal")
axs[3].set_title("targets")
# axs[2].set_aspect("equal")
axs[3].axis("off")

plt.tight_layout()
plt.savefig("medsiglip.png", bbox_inches="tight", dpi=300)

targets = torch.stack(targets).cuda()

try:
	for i, a in enumerate([logits1, logits2, logits3, targets]):
		torch.save(a, f=f"logits{i}.pt")
except Exception as e:
	print(e)

print("\n--cross entropy--")
from torch.functional import F

print(f"logits1: {F.binary_cross_entropy_with_logits(logits1, targets)}")
print(f"logits2: {F.binary_cross_entropy_with_logits(logits2, targets)}")
print(f"logits3: {F.binary_cross_entropy_with_logits(logits3, targets)}")

for i in range(len(targets)):
	print(F.binary_cross_entropy_with_logits(logits1[i], targets[i]))

sys.exit()

checkpoint = 'xray_04_16_003915_2.pt'
weights = torch.load(f"{xray_cnn.checkpoints_dir}/{checkpoint}", weights_only=True, map_location="cpu")
model = xray_cnn.XrayModel.load_weights(xray_cnn.XrayModel(), weights).to(device)
model.eval()
with torch.no_grad():
	cnn_outputs = model(torch.stack(imgs).to(device=device, dtype=torch.float32) / 255., xray_view=torch.stack(views).to(device))
# cnn_outputs = torch.softmax(cnn_outputs, dim=1)
# Source - https://stackoverflow.com/a/47483819
# Posted by user1767754, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-21, License - CC BY-SA 4.0

torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)

for i in range(8):
	print(targets[i])
	print(probs[i])
	print(cnn_outputs[i])
	print()
...
