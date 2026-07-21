import collections

from torch.utils.data import DataLoader
from tqdm import tqdm

from training.xray.xray_data import *

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")


def compute_class_frequency():
	total = len(annotations.index)
	sums = collections.OrderedDict()

	for col in annotations.columns[3:17]:
		sums[col] = annotations[col].sum().item()

	return sums, total


def find_sickest_patient():
	max_findings, finding = 0, None
	for row in annotations.iloc:
		findings = row.iloc[3:17].sum().item()
		if findings > max_findings:
			max_findings, finding = findings, row['dicom_id']
	return max_findings, finding


def find_labelless_entries():
	empty_row_numbers = []
	for row in annotations.iloc:
		findings = row.iloc[3:17].sum().item()
		if findings == 0:
			empty_row_numbers.append(row.name)
	return empty_row_numbers


def compute_mean_and_variance():
	transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
	dataset = XrayDataset(transform=transform)
	loader = DataLoader(dataset, batch_size=64, shuffle=False)

	pixel_sum = None
	pixel_squared_sum = None
	pixel_count = 0

	for (imgs, _), _ in tqdm(loader):
		imgs = imgs.to(torch.float64)
		reduce_dims = (0, *range(2, imgs.ndim))

		batch_sum = imgs.sum(dim=reduce_dims)
		batch_squared_sum = (imgs * imgs).sum(dim=reduce_dims)
		batch_count = imgs.shape[0] * imgs[0, 0].numel()

		if pixel_sum is None:
			pixel_sum = batch_sum
			pixel_squared_sum = batch_squared_sum
		else:
			pixel_sum += batch_sum
			pixel_squared_sum += batch_squared_sum

		pixel_count += batch_count

	mean = pixel_sum / pixel_count
	variance = pixel_squared_sum / pixel_count - mean.square()

	return mean.item(), variance.item()


if __name__ == '__main__':
	metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
	annotations = pd.read_csv(f"{data_dir}/annotations.csv")

# print(compute_class_frequency())
# print(find_sickest_patient())
# mean, variance = compute_mean_and_variance()
# print(f"Mean: {mean}, Variance: {variance}")
# labelless = find_labelless_entries()
# print(len(labelless))
# print(labelless[0:min(len(labelless), 16)])
# print(labelless[-min(len(labelless), 16):])
