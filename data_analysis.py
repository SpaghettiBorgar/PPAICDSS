import collections
import os

import pandas as pd

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")


def compute_class_frequency():
	total = len(annotations.index)
	sums = collections.OrderedDict()

	for col in annotations.columns[3:17]:
		sums[col] = annotations[col].sum().item()

	return sums, total

if __name__ == '__main__':
	metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
	annotations = pd.read_csv(f"{data_dir}/annotations.csv")

	print(compute_class_frequency())

class_weights = collections.OrderedDict({'Atelectasis': 36574, 'Cardiomegaly': 34348, 'Consolidation': 7353, 'Edema': 19519, 'Enlarged Cardiomediastinum': 5317, 'Fracture': 3604, 'Lung Lesion': 6418, 'Lung Opacity': 43730, 'No Finding': 89070, 'Pleural Effusion': 41462, 'Pleural Other': 1748, 'Pneumonia': 14726, 'Pneumothorax': 7547, 'Support Devices': 42806})
total_samples = 221121