import ollama
import os
import json
import sys

from pathlib import Path
from dotenv import load_dotenv

def get_response(stream):
    full_response = ""

    for chunk in stream:
        content = chunk['message']['content']
        full_response += content
    
    return full_response

def ask_ollama(prompt: str, task: str, llm_config: dict = None):
    # get llm config
    model = llm_config.get("model", "gemma3:4b")
    if llm_config["max_output_tokens"] is None:
        max_output_tokens = 512
    else:
        max_output_tokens = llm_config.get("max_output_tokens", 512).get(task, 512)
    
    temperature = llm_config.get("temperature", 0.2)
    
    # get prompt
    tmp = prompt.split("[input]")
    system_prompt = tmp[0]
    user_prompt = tmp[1]

    # Ask Gemma
    try:
        stream = ollama.chat(
            model = model,
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': temperature,
                'num_predict':max_output_tokens,
            },
            stream=True
        )
        
        res = get_response(stream)

        return res
    except ollama.ResponseError as e:
        print(f"錯誤：{e.error}")
        if "model" in e.error and "not found" in e.error:
            print(f"錯誤提示：請確認模型名稱 '{model}' 是否正確，")
            print("您可以執行 'ollama list' 來查看所有已安裝的模型。")
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
    
    return ""