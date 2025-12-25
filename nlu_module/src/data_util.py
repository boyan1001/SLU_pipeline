import json
import os

def load_json(file_path):
    """Load a JSON file and return its content as a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return None

def save_json(data, file_path):
    """Save a dictionary to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def gets_slot_list(dir_path):
    """Retrieve the list of SNIPS slots from a JSON file."""
    slot_list = []

    for file_name in os.listdir(dir_path):
        if not file_name.endswith('.json'):
            continue
        if file_name == "intent_list.json" or file_name == "slot_list.json":
            continue

        file_path = os.path.join(dir_path, file_name)
        raw_data = load_json(file_path)
        if raw_data is None:
            continue

        for item in raw_data:
            slots_line = item.get("slots", "")
            slots = slots_line.split()
            utterance = item.get("utterance", "")
            words = utterance.split()
            
            if len(slots) != len(words):
                print(f"Warning: Mismatch in lengths for utterance '{utterance}'")
                continue

            for i in range(len(slots)):
                slot = slots[i]
                word = words[i]

                if slot == "O":
                    continue
                slot = slot.split("-")[-1]
                if slot not in [s["slot_name"] for s in slot_list]:
                    slot_data = {
                        "slot_name": slot,
                        "description": "",
                        "examples": []
                    }
                    slot_list.append(slot_data)
                if file_name == "train.json":
                    slot_list_index = next(index for (index, d) in enumerate(slot_list) if d["slot_name"] == slot)
                    if word not in slot_list[slot_list_index]["examples"] and slot_list[slot_list_index]["examples"].__len__() < 10:
                        slot_list[slot_list_index]["examples"].append(word)
                
    return slot_list

def get_intent_list(dir_path):
    """Retrieve the list of ATIS intents from a JSON file."""
    intent_list = []

    for file_name in os.listdir(dir_path):
        if not file_name.endswith('.json'):
            continue
        if file_name == "intent_list.json" or file_name == "slot_list.json":
            continue

        file_path = os.path.join(dir_path, file_name)
        raw_data = load_json(file_path)
        if raw_data is None:
            continue

        for item in raw_data:
            intent = item.get("intent", "")
            utterance = item.get("utterance", "")

            if intent not in [i["intent_name"] for i in intent_list]:
                intent_data = {
                    "intent_name": intent,
                    "description": "",
                    "examples": []
                }
                intent_list.append(intent_data)
            if file_name == "train.json":
                intent_list_index = next(index for (index, d) in enumerate(intent_list) if d["intent_name"] == intent)
                if utterance not in intent_list[intent_list_index]["examples"] and intent_list[intent_list_index]["examples"].__len__() < 10:
                    intent_list[intent_list_index]["examples"].append(utterance)

    return intent_list

def count_json_length(file_path):
    """Count the number of entries in a JSON file."""
    data = load_json(file_path)
    if data is not None:
        return len(data)
    return 0

if __name__ == "__main__":
    dir_path = "./data/SLURP"
    slot_list = gets_slot_list(dir_path)
    intent_list = get_intent_list(dir_path)
    print("Slots:", slot_list)
    print("Intents:", intent_list)
    
    save_json(slot_list, os.path.join(dir_path, "slot_list.json"))
    save_json(intent_list, os.path.join(dir_path, "intent_list.json"))

