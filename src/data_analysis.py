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
