import os
import json
import time
import multiprocessing as mp
import csv

import src.bio_seq as bs
import src.metrics as me
import src.slot_norm as sn
import model.providers as pr

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from model.providers import ask
from model.providers import init_worker as providers_init_worker

DEBUG = False


def load_prompt_template(template_path: Path):
    with open(template_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data


def _worker_process_item(args):
    item, dataset, slot_des, slot_exp, llm_config, debug, stage = args
    utterance = item.get("utterance", "")
    if not utterance:
        return None

    try:
        resp = gpt_slu(utterance, dataset, slot_des=slot_des, slot_exp=slot_exp, llm_config=llm_config, debug=debug, stage=stage)
        pred = {
            "utterance": utterance,
            "predicted_intent": resp["intent"],
            "predicted_slots": resp["slots"],
            "true_intent": item.get("intent", ""),
            "true_slots": item.get("slots", ""),
            "response": resp["response"]
        }

        # 局部統計
        true_slots_seq = item.get("slots","")
        if isinstance(true_slots_seq, list):
            true_slots_seq = " ".join(true_slots_seq)
        true_slots_seq = true_slots_seq.split()

        pred_slots_seq = resp.get("slots", "")
        if isinstance(pred_slots_seq, list):
            pred_slots_seq = " ".join(pred_slots_seq)
        pred_slots_seq = pred_slots_seq.split()

        pred_slots_seq = bs.pad_or_trim(pred_slots_seq, len(true_slots_seq))

        return pred, 1, pred_slots_seq, true_slots_seq

    except Exception as e:
        import traceback, sys
        print(f"[worker error] utterance={utterance!r} error={e}", file=sys.stderr)
        traceback.print_exc()
        return None

# NOTE: Rate-limiting and provider-specific client initialization are
# delegated to the provider modules. We keep a small compatibility shim
# above that exposes `ask_gpt`.
# ================
# GPT-SLU Main Functions
# ================


def intent_detection(input: str, dataset:str, stage: int, maybe_intent: str = "", maybe_slot: str = "", llm_config: dict = None):
    # Find intent list
    intent_list_path = Path(__file__).parent.parent / "data" / dataset / "intent_list.json"
    try:
        with open(intent_list_path, 'r', encoding="utf-8") as file:
            intent_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Intent list file not found at {intent_list_path}")

    intent_list = [intent["intent_name"] for intent in intent_data]
    intent_list_str = ", ".join(intent_list)

    # Design prompt
    if stage == 1:
        intent_prompt = f"""
        [schema]
        Given the following sentences, choose the intent of annotations the sentences from the following intent list:
        [{intent_list_str}]

        [regulation]
        - Output the annotations in the form of [The intent is: <intent_name>]
        - must not output anything else other than the extracted information
        - intent MUST be chosen from intent list

        [example]
        utterance: "A B C D E F G"
        intent: "[The intent is: H]"

        [input]
        The input sentences is: [{input}]
        """
    elif stage == 2:
        intent_prompt = f"""
        [schema]
        Given the following sentences, [{maybe_slot}] may be the slots, choose the intent of annotations the sentences from the following intent list:
        [{intent_list_str}]

        [regulation]
        - Output the extracted information in the form of [The intent is: <intent_name>]
        - must not output anything else other than the extracted information
        - intent MUST be chosen from intent list

        [example]
        utterance: "A B C D E F G"
        intent: "[The intent is: H]"

        [input]
        The input sentences is: [{input}]
        """
    else:
        intent_prompt = f"""
        [schema]
        Given the following sentences, [{maybe_intent}] may be the intent; and [{maybe_slot}] may be the slots.
        Choose the intent of annotations the sentences from the following intent list:
        [{intent_list_str}]

        [regulation]
        - Output the extracted information in the form of [The intent is: <intent_name>]
        - must not output anything else other than the extracted information
        - intent MUST be chosen from intent list

        [example]
        utterance: "A B C D E F G"
        intent: "[The intent is: H]"

        [input]
        The input sentences is: [{input}]
        """
    
    # print("--- Intent prompt ----")
    # print(intent_prompt)
    ans = pr.ask(intent_prompt, task="intent_detection", llm_config=llm_config)

    # Defensive: ensure ans is a string (provider may return None on error)
    if ans is None:
        return ""
    if not isinstance(ans, str):
        ans = str(ans)

    return ans


def slot_filling(input: str, dataset:str, stage: int, maybe_intent: str = "", maybe_slot: str = "", slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None):
    # Find slot list
    slot_list_path = Path(__file__).parent.parent / "data" / dataset / "slot_list.json"
    try:
        with open(slot_list_path, 'r', encoding="utf-8") as file:
            slot_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Intent list file not found at {slot_list_path}")
    
    # slot list
    slot_list = []
    if slot_des and slot_exp:
        for slot in slot_data:
            slot_list.append(f"{slot['slot_name']} (Description: {slot.get('description', 'N/A')}; Example: {slot.get('example', 'N/A')})")
    elif slot_des:
        for slot in slot_data:
            slot_list.append(f"{slot['slot_name']} (Description: {slot.get('description', 'N/A')})")
    elif slot_exp:
        for slot in slot_data:
            slot_list.append(f"{slot['slot_name']} (Example: {slot.get('examples', 'N/A')})")
    else:
        slot_list = [slot["slot_name"] for slot in slot_data]

    slot_list_str = ", ".join(slot_list)

    # Design prompt
    if stage == 1:
        slot_prompt = f"""
        [schema]
        Given the following sentences, choose the slots of annotations the sentences from the following slot list:
        [{slot_list_str}]

        [regulation]
        - Each line MUST contain EXACTLY ONE pair: [<value> -> <slot_name>]
        - Do NOT put more than one "->" inside a single pair.
        - Do NOT group multiple pairs inside the same brackets.
        For example, this is FORBIDDEN:
        [zapata -> restaurant_name, four -> party_size_number]
        - If you need multiple pairs, write them on SEPARATE lines:
        [zapata -> restaurant_name]
        [four -> party_size_number]
        - slot_name MUST be chosen ONLY from: [...]
        - Do NOT invent new slot names.
        - Do NOT add extra words (like "slot", "entity", etc.) to the slot_name.

        [example]
        utterance: "A B C D E F G"
        slots: "
        [<value1> -> <slot1>]
        [<value2> -> <slo12>]
        "
        
        [input]
        The input sentences is: [{input}]
        """
    elif stage == 2:
        slot_prompt = f"""
        [schema]
        Given the following sentences, [{maybe_intent}] may be the intent, choose the slots of annotations the sentences from the following slot list:
        [{slot_list_str}]

        [regulation]
        - Each line MUST contain EXACTLY ONE pair: [<value> -> <slot_name>]
        - Do NOT put more than one "->" inside a single pair.
        - Do NOT group multiple pairs inside the same brackets.
        For example, this is FORBIDDEN:
        [zapata -> restaurant_name, four -> party_size_number]
        - If you need multiple pairs, write them on SEPARATE lines:
        [zapata -> restaurant_name]
        [four -> party_size_number]
        - slot_name MUST be chosen ONLY from: [...]
        - Do NOT invent new slot names.
        - Do NOT add extra words (like "slot", "entity", etc.) to the slot_name.

        [example]
        utterance: "A B C D E F G"
        slots: "
        [<value1> -> <slot1>]
        [<value2> -> <slo12>]
        "

        [input]
        The input sentences is: [{input}]
        """
    else:
        slot_prompt = f"""
        [schema]
        Given the following sentences, [{maybe_intent}] may be the intent; and [{maybe_slot}] may be the slots.
        Choose the slots of annotations the sentences from the following slot list:
        [{slot_list_str}]

        [regulation]
        - Each line MUST contain EXACTLY ONE pair: [<value> -> <slot_name>]
        - Do NOT put more than one "->" inside a single pair.
        - Do NOT group multiple pairs inside the same brackets.
        For example, this is FORBIDDEN:
        [zapata -> restaurant_name, four -> party_size_number]
        - If you need multiple pairs, write them on SEPARATE lines:
        [zapata -> restaurant_name]
        [four -> party_size_number]
        - slot_name MUST be chosen ONLY from: [...]
        - Do NOT invent new slot names.
        - Do NOT add extra words (like "slot", "entity", etc.) to the slot_name.

        [example]
        utterance: "A B C D E F G"
        slots: "
        [<value1> -> <slot1>]
        [<value2> -> <slo12>]
        "

        [input]
        The input sentences is: [{input}]
        """
    
    # print("--- slot filling ---")
    # print(slot_prompt)
    ans = pr.ask(slot_prompt, task="slot_filling", llm_config=llm_config)

    if ans is None:
        return ""
    if not isinstance(ans, str):
        ans = str(ans)

    return ans

def stage_one(input: str, dataset: str, slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None):
    pred_intent = intent_detection(
        input, dataset, stage=1, llm_config=llm_config
    )

    if not pred_intent:
        pred_intent = ""
    else:
        pred_intent = pred_intent.split(":")[-1].strip().strip("]").strip()

    pred_slots = slot_filling(
        input, dataset, stage=1, slot_des=slot_des, slot_exp=slot_exp, llm_config=llm_config
    )
    return pred_intent, pred_slots

def stage_two(input: str, dataset: str, pred_intent: str, pred_slots: str, slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None):
    refined_intent = intent_detection(
        input, dataset, stage=2, maybe_slot=pred_slots, llm_config=llm_config
    )
    if not refined_intent:
        refined_intent = ""
    else:
        refined_intent = refined_intent.split(":")[-1].strip().strip("]").strip()

    refined_slots = slot_filling(
        input, dataset, stage=2, maybe_intent=pred_intent, slot_des=slot_des, slot_exp=slot_exp, llm_config=llm_config
    )
    return refined_intent, refined_slots

def stage_three(input: str, dataset: str, pred_intent: str, pred_slots: str, slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None):
    refined_intent = intent_detection(
        input, dataset, stage=3, maybe_intent=pred_intent, maybe_slot=pred_slots, llm_config=llm_config
    )
    if not refined_intent:
        refined_intent = ""
    else:
        refined_intent = refined_intent.split(":")[-1].strip().strip("]").strip()

    refined_slots = slot_filling(
        input, dataset, stage=2, maybe_intent=pred_intent, maybe_slot=pred_slots, slot_des=slot_des, slot_exp=slot_exp, llm_config=llm_config
    )
    return refined_intent, refined_slots

def gpt_slu(utterance: str, dataset: str = "", slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None, debug: bool = False, stage: int = 2):
    # === Stage 1 ===
    s1_intent, s1_slots = stage_one(utterance, dataset, slot_des, slot_exp, llm_config)

    if debug:
        print("=== stage 1 ===")
        print("--- Predicted intent ---")
        print(s1_intent)
        print("\n--- Predicted slots ---")
        print(s1_slots)

    s1_slots = sn.normalize_multi_pairs(s1_slots)

    if stage == 1:
        s1_slots_seq = bs.pairs_to_bio_seq(utterance, s1_slots)
        return {
            "utterance": utterance,
            "slots": s1_slots_seq,
            "intent": s1_intent,
            "response": {
                "stage 1":{
                    "intent": s1_intent,
                    "slots": s1_slots,
                }
            }
        }
    
    # === Stage 2 ===
    s1_slots = bs.format_slots_hint(s1_slots)
    s2_intent, s2_slots = stage_two(utterance, dataset, s1_intent, s1_slots, slot_des, slot_exp, llm_config)
   
    if debug:
        print("\n=== stage 2 ===")
        print("--- Predicted intent ---")
        print(s2_intent)
        print("\n--- Predicted slots ---")
        print(s2_slots)

    s2_slots = sn.normalize_multi_pairs(s2_slots)

    if stage == 2:
        s2_slots_seq = bs.pairs_to_bio_seq(utterance, s2_slots)
        return {
            "utterance": utterance,
            "slots": s2_slots_seq,
            "intent": s2_intent,
            "response": {
                "stage 1":{
                    "intent": s1_intent,
                    "slots": s1_slots,
                },
                "stage 2":{
                    "intent": s2_intent,
                    "slots": s2_slots
                }
            }
        }

    # === Stage 3 ===
    s2_slots = bs.format_slots_hint(s2_slots)
    s3_intent, s3_slots = stage_three(utterance, dataset, s2_intent, s2_slots, slot_des, slot_exp, llm_config)

    if debug:
        print("\n=== stage 3 ===")
        print("--- Predicted intent ---")
        print(s3_intent)
        print("\n--- Predicted slots ---")
        print(s3_slots)

    s3_slots = sn.normalize_multi_pairs(s3_slots)
    s3_slots_seq = bs.pairs_to_bio_seq(utterance, s3_slots)

    return {
        "utterance": utterance,
        "slots": s3_slots_seq,
        "intent": s3_intent,
        "response": {
            "stage 1":{
                "intent": s1_intent,
                "slots": s1_slots,
            },
            "stage 2":{
                "intent": s2_intent,
                "slots": s2_slots
            },
            "stage 3":{
                "intent": s3_intent,
                "slots": s3_slots
            }
        }
    }

def batch_gpt_slu(dataset: str, output_path: str, slot_des: bool = False, slot_exp: bool = False, llm_config: dict = None, debug: bool = False, stage: int = 2, mode:str = 'all'):
    # Prepare input and output paths
    input_dir_path = Path(__file__).parent.parent / "data" / dataset
    if output_path == "":
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"gpt_slu_{dataset}_{now_str}"
        output_file_name = f"{exp_name}.json"
        output_path = Path(__file__).parent.parent / "output" / output_file_name

    # Load datasets
    predictions = []
    raw_data = []
    
    for file_name in os.listdir(input_dir_path):
        if mode == "all":
            if file_name not in ("test.json"):
                continue
        if mode == "test":
            if file_name != "test2.json":
                continue
        file_path = "/datas/store162/boyan/code/gpt-slu/test_bias.json"
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                raw_data.extend(json.load(f))
        except Exception:
            continue

    # Ratec limit and concurrency settings
    provider_name = llm_config.get("provider", "gemini")
    model_name = llm_config.get("model", "gemini-2.5-flash")

    if provider_name in ["ollama"]:
        rpm = 1000000
        concurrency = 3
    else:
        rpm = int(os.getenv("OPENAI_RPM", "60"))
        concurrency = int(os.getenv("OPENAI_CONCURRENCY", "5"))

    # 建立共享限流資源
    manager = mp.Manager()
    shared_ns = manager.Namespace()
    shared_ns.rpm = rpm
    shared_ns.window_start = time.time()
    shared_ns.count = 0

    lock = manager.Lock()
    sem = manager.BoundedSemaphore(concurrency)

    # 多進程池
    cpu_workers = max(1, mp.cpu_count() - 1)  # 留一顆避免把機器榨乾
    pool = mp.Pool(
        processes=cpu_workers,
        initializer=providers_init_worker,
        initargs=(shared_ns, lock, sem, provider_name, model_name)
    )

    # 建立任務
    tasks = ((item, dataset, slot_des, slot_exp, llm_config, debug, stage) for item in raw_data)

    # 統計
    dataset_size = len(raw_data)

    # calculate consume time
    start_time = time.time()
    pred_intent_all = []
    true_intent_all = []
    pred_seq_all = []
    true_seq_all = []
    processed_count = 0
    failed_count = 0
    try:
        with tqdm(total=dataset_size, desc="Processing utterances (mp)", leave=False) as bar:
            for result in pool.imap_unordered(_worker_process_item, tasks):
                processed_count += 1
                if result is None:
                    failed_count += 1
                    bar.update(1)
                    continue
                pred, tot, pred_slots_seq, true_slots_seq = result
                predictions.append(pred)
                pred_intent_all.append(pred["predicted_intent"])
                true_intent_all.append(pred["true_intent"])
                pred_seq_all.append(pred_slots_seq)
                true_seq_all.append(true_slots_seq)
                bar.update(1)
    finally:
        pool.close()
        pool.join()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # turn format to hh:mm:ss
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # 存檔
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(predictions, out_file, ensure_ascii=False, indent=4)

    # Metrics
    metrics = me.get_metrics(pred_intent_all, true_intent_all, pred_seq_all, true_seq_all)
    intent_acc = metrics["intent_acc"]
    total_acc = metrics["overall_acc"]
    slot_prec = metrics["slot_prec"]
    slot_rec = metrics["slot_rec"]
    slot_f1 = metrics["slot_f1"]
    invalid = metrics["invalid"]
    
    dataset_size -= invalid

    record_file = Path(__file__).parent.parent / "output" / "record.csv"
    response = {
        "exp_name": exp_name,
        "model_name": model_name,
        "num_stages": stage,
        "dataset": dataset,
        "dataset_size": dataset_size,
        "time": elapsed_time_str,
        "intent_acc": intent_acc,
        "slot_prec": slot_prec,
        "slot_rec": slot_rec,
        "slot_f1": slot_f1,
        "total_acc": total_acc,
    }
    try:
        with open(record_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            add_info = [
                exp_name,
                model_name,
                stage,
                dataset,
                dataset_size,
                elapsed_time_str,
                intent_acc,
                slot_prec,
                slot_rec,
                slot_f1,
                total_acc
            ]
            writer.writerow(add_info)
    except Exception:
        pass
    
    return response
