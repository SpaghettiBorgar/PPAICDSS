from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForMultimodalLM

from cdss.tools import get_weather
from models import Tool

# MODEL_ID = "google/medgemma-4b-it"
MODEL_ID = "google/gemma-4-E2B-it"

SYSTEM_PROMPT = """
You are a medical assistant.

You have access to these tools:

Tool: search_pubmed
Description: Search medical literature database.

When you need a tool, respond ONLY in valid JSON:

{
  "tool": "search_pubmed",
  "arguments": {
      "query": "..."
  }
}

If no tool is needed, answer normally.
"""


def test():
	model = init_chat_model(MODEL_ID, model_provider="huggingface", temperature=0.7, max_tokens=1024)
	tools = [get_weather]
	model.bind_tools(tools)
	rendered_tools = render_text_description(tools)
	print(rendered_tools)

	# system_prompt = f"""\
	# You are an assistant that has access to the following set of tools.
	# Here are the names and descriptions for each tool:
	#
	# {rendered_tools}
	#
	# Given the user input, return the name and input of the tool to use.
	# Return your response as a JSON blob with 'name' and 'arguments' keys.
	#
	# The `arguments` should be a dictionary, with keys corresponding
	# to the argument names and the values corresponding to the requested values.
	# """

	system_prompt = "You are a helpful assistant"

	prompt = ChatPromptTemplate.from_messages(
		[("system", system_prompt), ("user", "{input}")]
	)
	chain = prompt | model
	message = chain.invoke({"input": "What's the weather like in Boston?"})

	# Let's take a look at the output from the model
	# if the model is an LLM (not a chat model), the output will be a string.
	if isinstance(message, str):
		print(message)
	else:  # Otherwise it's a chat model
		print(message.content)

	response = model.invoke("What's the weather like in Boston?")
	print(response)
	for tool_call in response.tool_calls:
		# View tool calls made by the model
		print(f"Tool: {tool_call['name']}")
		print(f"Args: {tool_call['args']}")


class ToolCall(BaseModel):
	tool: str
	arguments: dict


class MedGemma:

	def __init__(self, device_map="auto"):
		self.model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map=device_map)
		self.processor = AutoProcessor.from_pretrained(MODEL_ID)
		self.tools = []
		self.json_parser = JsonOutputParser(pydantic_object=ToolCall)

	def register_tool(self, *tools: Tool):
		self.tools.extend(tools)

	def generate(self, conversation, streamer=None, max_new_tokens=500):
		text = self.processor.apply_chat_template(
			conversation, tools=[t.schema for t in self.tools],
			tokenize=False, add_generation_prompt=True)
		inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
		outputs = self.model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
		return outputs
