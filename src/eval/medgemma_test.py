import os
from threading import Thread
from typing import List
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from transformers import (
	AutoModelForCausalLM,
	AutoModelForImageTextToText,
	AutoProcessor,
	AutoTokenizer,
	BitsAndBytesConfig,
	TextIteratorStreamer,
)

from training.xray import xray_data

MODEL_VARIANT = "medgemma-27b-it"
MODEL_ID = f"google/{MODEL_VARIANT}"

USE_QUANTIZATION = False
IS_THINKING = False

PROMPT = f"For the following labels, give a list of which are present in this Chest X-Ray. Only provide a comma-seperated list of labels that apply, don't say anything else. If necessary encapsulate labels in parantheses to signal low confidence.\nLabels: {xray_data.LABELS}\n"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
ROLE_INSTRUCTION = "You are an expert radiologist."

THINKING_VARIANTS = {
	"medgemma-1.5-4b-it",
	"medgemma-27b-it",
	"medgemma-27b-text-it",
}


def validate_config() -> bool:
	if IS_THINKING and MODEL_VARIANT not in THINKING_VARIANTS:
		print(
			"Note: Thinking is enabled for a non-thinking variant. "
			"Setting thinking to False."
		)
		return False

	return IS_THINKING


def build_model_kwargs() -> dict:
	kwargs = {
		"dtype": torch.bfloat16,
		"device_map": "auto",
	}

	if USE_QUANTIZATION:
		kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

	return kwargs


def load_model_and_processor():
	model_kwargs = build_model_kwargs()

	if "text" in MODEL_VARIANT:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
		model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
		return model, tokenizer, tokenizer

	processor = AutoProcessor.from_pretrained(MODEL_ID)
	model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
	return model, processor, processor.tokenizer


def download_image(image_url: str) -> str:
	filename = os.path.basename(urlparse(image_url).path)

	if os.path.exists(filename):
		return filename

	response = requests.get(image_url, timeout=30)
	response.raise_for_status()

	with open(filename, "wb") as image_file:
		image_file.write(response.content)

	return filename


def build_messages(prompt: str, images: List[Image.Image] | None, is_thinking: bool) -> list[dict]:
	if is_thinking:
		system_instruction = (
			f"SYSTEM INSTRUCTION: think silently if needed. {ROLE_INSTRUCTION}"
		)
	else:
		system_instruction = ROLE_INSTRUCTION

	user_content = [{"type": "text", "text": prompt}]

	if images is not None:
		for image in images:
			user_content.append({"type": "image", "image": image})

	return [
		{
			"role": "system",
			"content": [{"type": "text", "text": system_instruction}],
		},
		{
			"role": "user",
			"content": user_content,
		},
	]


def prepare_inputs(processor, messages: list[dict], model):
	inputs = processor.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_dict=True,
		return_tensors="pt",
	)

	return inputs.to(model.device, dtype=torch.bfloat16)


def stream_response(model, tokenizer, inputs, max_new_tokens: int) -> str:
	streamer = TextIteratorStreamer(
		tokenizer,
		skip_prompt=True,
		skip_special_tokens=True,
	)

	generation_kwargs = {
		**inputs,
		"streamer": streamer,
		"max_new_tokens": max_new_tokens,
		"do_sample": False,
	}

	generation_thread = Thread(
		target=model.generate,
		kwargs=generation_kwargs,
	)

	generation_thread.start()

	chunks: list[str] = []

	for chunk in streamer:
		print(chunk, end="", flush=True)
		chunks.append(chunk)

	generation_thread.join()

	return "".join(chunks)


def split_thinking_trace(response: str) -> tuple[str | None, str]:
	if "<unused95>" not in response:
		return None, response

	thought, final_response = response.split("<unused95>", maxsplit=1)
	thought = thought.replace("<unused94>thought\n", "").strip()

	return thought, final_response.strip()


def main() -> None:
	is_thinking = validate_config()
	max_new_tokens = 1300 if is_thinking else 1200

	model, processor, tokenizer = load_model_and_processor()

	for img in ["xray1.jpg", "xray2.jpg", "xray3.jpg", "xray4.jpg", "xray5.jpg", "xray6.jpg", "xray7.jpg", "xray8.jpg", "sickest_patient.jpg", "rib_fracture.jpg"]:
		print(img)
		images = []
		if "text" not in MODEL_VARIANT:
			# image_filename = download_image(IMAGE_URL)
			image_filenames = [img]
			images = [Image.open(image_filename).convert("RGB") for image_filename in image_filenames]

		messages = build_messages(PROMPT, images, is_thinking)
		inputs = prepare_inputs(processor, messages, model)

		# print(f"---\n\n**[ User ]**\n\n{PROMPT}\n")
		print(messages)
		print("---\n\n**[ MedGemma ]**\n\n", end="", flush=True)

		with torch.inference_mode():
			response = stream_response(
				model=model,
				tokenizer=tokenizer,
				inputs=inputs,
				max_new_tokens=max_new_tokens,
			)

		if is_thinking:
			thought, final_response = split_thinking_trace(response)

			if thought:
				print(f"\n\n---\n\n**[ MedGemma thinking trace ]**\n\n{thought}")

			print(f"\n\n---\n\n**[ MedGemma final response ]**\n\n{final_response}")

		print("\n\n---")


if __name__ == "__main__":
	main()
