import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

def ask_gemini(prompt: str, task: str, llm_config: dict = None):
    # 找 .env 檔案並載入變數（確保在當前工作目錄和專案根目錄都找）
    for env_path in [Path(".env"), Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break

    # API key handling
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key and llm_config:
        api_key = llm_config.get("api_key")
    if not api_key:
        print("[error] GENAI_API_KEY not found in env or llm_config", file=sys.stderr)
        return None

    # get llm config
    model = llm_config.get("model", "gemini-2.0-flash")
    if llm_config["max_output_tokens"] is None:
        max_output_tokens = 512
    else:
        max_output_tokens = llm_config.get("max_output_tokens", 512).get(task, 512)
    
    temperature = llm_config.get("temperature", 0.2)
    # """
    # [schema]
    # Given the following sentences, [{maybe_intent}] may be the intent, choose the slots of annotations the sentences from the following slot list:
    # [{slot_list_str}]

    # [regulation]
    # - Output the extracted information in the form of [<value> -> <slot_name>]
    # - must not output anything else other than the extracted information

    # [example]
    # utterance: "A B C D E F G"
    # slots: "
    # [A -> slot A]
    # [B C -> slot B]
    # "

    # [input]
    # The input sentences is: [{input}]
    # """

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    tmp = prompt.split("[input]")
    system_prompt = tmp[0]
    user_prompt = tmp[1]

    # Ask Gemini model
    response = client.models.generate_content(
        model = model,
        contents = user_prompt,
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature = temperature,
            max_output_tokens = max_output_tokens,
        )
    )

    return response.text