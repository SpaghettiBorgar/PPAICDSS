from transformers.utils import get_json_schema

MODEL_ID = "google/gemma-4-E2B-it"
# MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

from transformers import AutoProcessor, AutoModelForMultimodalLM, TextStreamer

import cdss.tools

tools = {f.__name__: f for f in [cdss.tools.get_weather]}
tool_schemas = [get_json_schema(t) for t in tools.values()]

model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)


def extract_tool_calls(text):
	import re
	def cast(v):
		try:
			return int(v)
		except:
			try:
				return float(v)
			except:
				return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

	return [{
		"name": name,
		"arguments": {
			k: cast((v1 or v2).strip())
			for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
		}
	} for name, args in re.findall(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL)]


prompt = "What's the temperature in London?"
messages = [
	{
		"role": "system", "content": "You are a helpful assistant."
	},
	{
		"role": "user", "content": prompt
	}
]

text = processor.apply_chat_template(messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=2000)
generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=False)

print(f"Prompt: {prompt}")
print(f"Tools: {tools}")
print(f"Output: {output}")

calls = extract_tool_calls(output)
if calls:
	results = [
		{"name": c['name'], "response": tools[c['name']](**c['arguments'])}
		for c in calls
	]

	messages.append({
		"role": "assistant",
		"tool_calls": [
			{"function": call} for call in calls
		],
		"tool_responses": results
	})
	print(messages[-1])

text = processor.apply_chat_template(messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
messages[-1]["content"] = output

print("-" * 80)
print("Full History")
print("-" * 80)
import json

print(json.dumps(messages, indent=2))
