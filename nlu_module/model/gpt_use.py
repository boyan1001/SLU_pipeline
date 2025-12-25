import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def ask_openai(prompt: str, task: str, llm_config: dict):
    # env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # config
    model = llm_config.get("model", "gpt-5-mini")
        
    if task == "intent_detection":
        max_output_tokens = llm_config.get("max_output_tokens", 512).get("intent_detection", 512)
    elif task == "slot_filling":
        max_output_tokens = llm_config.get("max_output_tokens", 512).get("slot_filling", 512)
    else:
        raise ValueError("Invalid task type. Must be 'intent_detection' or 'slot_filling'.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        input=prompt
    )

    return response.output_text