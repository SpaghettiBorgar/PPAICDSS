import inspect

import torch
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class VLM:
	def __init__(self, model_id, tools=None, system_prompt=None, dtype="auto", device_map="auto"):
		self.processor = AutoProcessor.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
		self.messages = []
		self.images = []
		self.tools = tools if tools is not None else []
		if system_prompt is not None:
			self.add_system_message(system_prompt)

	def add_tool(self, tool):
		if not tool in self.tools:
			self.tools.append(tool)

	def add_system_message(self, msg):
		self.messages.append({"role": "system", "content": [{"type": "text", "text": msg}]})

	def generate(self, *messages, max_new_tokens=1024, **kwargs):
		content = []
		for msg in messages:
			if isinstance(msg, dict):
				self.messages.append(msg)
			else:
				if isinstance(msg, str):
					content.append({"type": "text", "text": msg})
				elif isinstance(msg, Image.Image):
					self.images.append(img := msg)
					content.append({"type": "image", "image": img})
				elif isinstance(msg, torch.Tensor):
					self.images.append(img := torchvision.transforms.functional.to_pil_image(msg))
					content.append({"type": "image", "image": img})
				else:
					raise TypeError()
		if len(content) > 0:
			self.messages.append({"role": "user", "content": content})

		while True:
			inputs = self.processor.apply_chat_template(self.messages, tools=self.tools, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
			outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=max_new_tokens, **kwargs)
			out = self.processor.decode(outputs[0][len(inputs["input_ids"][0]):])
			response = self.processor.parse_response(out)
			if not 'content' in response:
				response['content'] = []
			if type(response['content']) is not list:
				response['content'] = [response['content']]
			for i, c in enumerate(response['content']):
				if type(c) is not dict:
					response['content'][i] = {"type": "text", "text": c}
			self.messages.append(response)

			if not 'tool_calls' in response:
				break
			for tool_call in response['tool_calls']:
				try:
					if not tool_call['type'] == 'function':
						raise Exception("Unknown tool_call type")
					fn_name, fn_args = tool_call['function']['name'], tool_call['function']['arguments']
					fn = next(tool for tool in self.tools if tool.__name__ == fn_name)
					fn_sig = inspect.signature(fn)
					fn_args = {
						arg: val if isinstance(val, ann) == ann else ann(val)
						for arg, val in fn_args.items()
						for ann in (fn_sig.parameters[arg].annotation,)
						if ann is not inspect._empty
					}
					print(f"Calling function {fn.__name__} with arguments {fn_args}")
					tool_result = fn(**fn_args)
					print(f"Result: {tool_result}")
				except StopIteration:
					tool_result = "Tool not found"
				except Exception as e:
					print(e)
					tool_result = f"Exception during tool call: {repr(e)}"
				finally:
					self.messages.append({'role': 'tool', 'content': [{"type": "text", "text": str(tool_result)}]})

		return '\n'.join([r['text'] for r in response['content'] if r['type'] == 'text'])
