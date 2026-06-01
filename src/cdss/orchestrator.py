from torchvision.transforms import v2

from cdss import tools
from cdss.tools import ImageContext
from models.vlm import VLM
from training.xray.xray_data import XrayDataset

MODEL_ID = "google/gemma-4-E2B-it"


def test():
	dataset = XrayDataset(use_chunks=False, transform=v2.ToPILImage())
	img_ctx = ImageContext()
	img_ctx.add(dataset[7][0][0], dataset[8][0][0])

	vlm = VLM(MODEL_ID)
	vlm.add_tool(img_ctx.bind(tools.analyze_xray))
	vlm.add_tool(tools.get_weather)

	vlm.add_system_message('''
		You are a radiologist helping medical practitioners diagnose and give treatment recommendations for patients.
		Testing mode. Be brief and skip any disclaimers.
		If any tools require an image as argument, you should use a zero-indexed integer to refer to the n-th image of the user's input (i.e. 0 for the first, 1 for the second image).
	''')

	response = vlm.generate("What possible conditions do these X-rays show?", *img_ctx.images)

	print(response)


if __name__ == '__main__':
	test()
