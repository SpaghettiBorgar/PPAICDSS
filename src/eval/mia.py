from models.xray_cnn import *
from training.xray.xray_data import *
from training.xray.xray_params import *
from util.mapping import tree_map
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def parse_args():
	parser = argparse.ArgumentParser(description="Train Xray Model")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use")
	parser.add_argument("--checkpoint", "-c", type=str, default="latest", help="Model checkpoint to load")
	parser.add_argument("--resolution", "-R", type=int, default=600, help="Max side length to scale images to")
	parser.add_argument("--crop", "-C", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=True, help="Crop/pad images to square")
	parser.add_argument("--num-samples", "-N", type=int, default=20000, help="Number of samples to test")

	return parser.parse_args()

def main():
	args = parse_args()

	params = XrayParams(checkpoint=args.checkpoint, device=args.device, resolution=args.resolution)
	model = params.get_model()
	model.eval()
	transform = params.get_transform()
	criterion = params.get_criterion()

	data_inc = XrayDataset(transform=transform, offset=0, size=args.num_samples)
	data_exc = XrayDataset(transform=transform, offset=-20000, size=args.num_samples)

	def test_samples(dataset, losses, n):
		for i in tqdm(range(n)):
			inp, target = dataset[i]
			inp, target = tree_map(lambda t: t.unsqueeze(0).to(params.device), (inp, target))
			pred = model(inp)
			loss = criterion(pred, target)
			losses.append(loss.item())
	
	print(f"Calculating {len(data_inc)} inclusion losses")
	inc_losses = []
	test_samples(data_inc, inc_losses, args.num_samples)
	print(f"Included loss mean: {sum(inc_losses) / len(inc_losses)}")

	print(f"Calculating {len(data_exc)} exclusion losses")
	exc_losses = []
	test_samples(data_exc, exc_losses, args.num_samples)
	print(f"Excluded loss mean: {sum(exc_losses) / len(exc_losses)}")

	print("Calculating auroc scores")
	auroc = roc_auc_score([0 for _ in range(len(inc_losses))] + [1 for _ in range(len(exc_losses))], inc_losses + exc_losses)

	print()
	print(f"MIA AUROC: {auroc}")

if __name__ == "__main__":
	main()