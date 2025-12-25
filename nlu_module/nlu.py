import typer
import json
import time

from pathlib import Path


def main(
        input: str = typer.Option("", "--input", "-i", help="Input line to process"),
        datasets: str = typer.Option("SNIPS", "--datasets", "-d", help="Datasets to process"),
        output_path: str = typer.Option("", "--output_path", "-o", help="Output path for processed data"),
        mode:str = typer.Option("all", "--mode", help="test / all"),
        slot_des: bool = typer.Option(False, "--slot_des", is_flag=True, help="Offer slot description"),
        slot_ex: bool = typer.Option(False, "--slot_ex", is_flag=True, help="Offer slot examples"),
        model: str = typer.Option("gemma3:12b", "--model", "-m", help="LLM model name"),
        debug: bool = typer.Option(False, "--debug", is_flag=True, help="Open debug mode"),
        stage: int = typer.Option(2, "--stage", "-s", help="Number of stages to use")
    ):
    if datasets not in ["ATIS", "SNIPS", "SLURP"]:
        raise ValueError("Invalid dataset specified. Choose from 'ATIS', 'SNIPS', or 'SLURP'.")
     
    import nlu_module.src.gpt_slu as gpt_slu
        
    llm_config_path = Path(__file__).parent / "config" / "llm_setting.json"
    
    try:
        with open(llm_config_path, 'r', encoding="utf-8") as file:
            llm_configs = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"LLM config file not found at {llm_config_path}")

    llm_config = None
    for llm in llm_configs:
        if llm.get("model", "") == model:
            llm_config = llm
            break
    if llm_config is None:
        raise ValueError(f"Model '{model}' not found in LLM config.")

    if input == "":
        response = gpt_slu.batch_gpt_slu(datasets, output_path, slot_des=slot_des, slot_exp=slot_ex, llm_config=llm_config, debug=debug, stage=stage, mode=mode)
        print("\n=== \033[1mFinal Result\033[0m ===")
        print("\033[36;1mModel: \033[0m", model)
        print("\033[36;1mDataset: \033[0m", datasets)
        print("\033[36;1mDataset Size: \033[0m", response["dataset_size"])
        print("\033[36;1mIntent Accuracy: \033[0m", response["intent_acc"])
        print("\033[36;1mSlot F1 Score: \033[0m", response["slot_f1"])
        print("\033[36;1mTotal Accuracy: \033[0m", response["total_acc"])
    else:
        start_time = time.time()
        response = gpt_slu.gpt_slu(input, datasets, slot_des=slot_des, slot_exp=slot_ex, llm_config=llm_config, debug=debug, stage=stage)
        end_time = time.time()
        latency = (end_time - start_time) * 1000

        print("\n=== \033[1mFinal Predicted Intent\033[0m ===")
        print("\033[36;1mModel: \033[0m", model)
        print("\033[36;1mDataset: \033[0m", datasets)
        print("\033[36;1mUtterance: \033[0m", response["utterance"])
        print("\033[36;1mIntent: \033[0m", response["intent"])
        print("\033[36;1mSlots: \033[0m", response["slots"])
        print("\033[36;1mLatency (ms): \033[0m", f"{latency:.2f}")


def nlu_pipeline(base_text: str, bias_text: str, datasets: str, model: str, stage: int):
    '''
    arg:{
        base_text: str,
        bias_text: str,
        datasets: str,
        model: str,
        stage: int
    }
    '''
    import nlu_module.src.gpt_slu as gpt_slu

    if datasets not in ["ATIS", "SNIPS", "SLURP"]:
        raise ValueError("Invalid dataset specified. Choose from 'ATIS', 'SNIPS', or 'SLURP'.")
    
    llm_config_path = "nlu_module/config/llm_setting.json"
    try:
        with open(llm_config_path, 'r', encoding="utf-8") as file:
            llm_configs = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"LLM config file not found at {llm_config_path}")
    
    llm_config = None
    for llm in llm_configs:
        if llm.get("model", "") == model:
            llm_config = llm
            break
    if llm_config is None:
        raise ValueError(f"Model '{model}' not found in LLM config.")

    if stage < 1 or stage > 3:
        raise ValueError("Stage must be 1, 2, or 3.")
    
    base_res = gpt_slu.gpt_slu(base_text, datasets, slot_des=False, slot_exp=False, llm_config=llm_config, debug=False, stage=stage)
    if bias_text != "":
        bias_res = gpt_slu.gpt_slu(bias_text, datasets, slot_des=False, slot_exp=False, llm_config=llm_config, debug=False, stage=stage)
        return base_res["intent"], base_res["slots"], bias_res["intent"], bias_res["slots"]
    else:
        return base_res["intent"], base_res["slots"]

if __name__ == "__main__":
    nlu_pipeline("test", "", "SNIPS", "gemma3:12b", 2)