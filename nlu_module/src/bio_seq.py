import re
import unicodedata

SLOT_PATTERNS = [
    r'\[([^][]+?)\s*->\s*([^\][]+?)\]',                           # [ value -> SLOT ]
    r'(?i)\b(.+?)\s+is\s+an?\s+([A-Za-z0-9_.-]+)\s+entity\b',     # value is a slot entity
    r'(?i)\bslot\s*[:=]\s*([A-Za-z0-9_.-]+)\s*,\s*value\s*[:=]\s*(.+?)$'
]

def normalize_slot_name(raw: str) -> str:
    s = raw.strip()
    # 把 "slot album" 這種前綴砍掉
    s = re.sub(r'(?i)^slot\s+', '', s)
    # 空白改底線
    s = re.sub(r'\s+', '_', s)
    # 保留安全字元
    s = re.sub(r'[^A-Za-z0-9_.-]', '_', s)
    if not s:
        s = 'misc'
    return s

def parse_slot_pairs(text: str):
    text = normalize_slot_name(text)
    pairs = []
    for pat in SLOT_PATTERNS:
        for m in re.finditer(pat, text):
            v, s = (m.group(1), m.group(2))
            v = v.strip(' ;,()[]{}"\'`').strip()
            s = s.strip(' ;,()[]{}"\'`').strip()
            if v and s: 
                pairs.append((v, s))
    # 長片段優先，避免重疊
    pairs = sorted({(v.lower(), s): (v, s) for v, s in pairs}.values(),
                   key=lambda x: -len(x[0]))
    return pairs

def normalize_unicode(s: str):
    return unicodedata.normalize("NFKC", s)

def tokenize_for_bio(text: str):
    text = normalize_unicode(text)
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def parse_value_slot_pairs(pred_text: str):
    pattern = r"\[([^][]+?)\s*->\s*([^\][]+?)\]"
    pairs = re.findall(pattern, pred_text)
    return [(v.strip(), s.strip()) for v, s in pairs if v.strip() and s.strip()]

def pairs_to_bio_seq(utterance: str, pred_text: str, slot_alias: dict = None):
    try:
        slot_alias = slot_alias or {}
        utter_tokens = tokenize_for_bio(utterance)
        L = len(utter_tokens)
        bio_tags = ["O"] * L
        used = [False] * L
        lower_utts = [w.lower() for w in utter_tokens]

        pairs = parse_value_slot_pairs(pred_text)

        # sort pairs by value length (longer phrase first)
        pairs = sorted(pairs, key=lambda x: -len(x[0]))

        for value, slot in pairs:
            slot = slot_alias.get(slot.lower(), slot)
            value_tokens = tokenize_for_bio(value)
            n = len(value_tokens)
            if n == 0:
                continue

            for i in range(L - n + 1):
                if lower_utts[i:i+n] == [w.lower() for w in value_tokens] and not any(used[i:i+n]):
                    # Label BIO
                    for k in range(n):
                        prefix = "B-" if k == 0 else "I-"
                        bio_tags[i+k] = prefix + slot
                        used[i+k] = True
                    break 
        return " ".join(bio_tags)
    except Exception:
        return "O " * len(tokenize_for_bio(utterance))

def format_slots_hint(pred_slots: str):
    pairs = parse_slot_pairs(pred_slots)
    return "; ".join([f"[{v} -> {s}]" for v, s in pairs])

def pad_or_trim(pred_seq, gold_len):
    if len(pred_seq) < gold_len:
        pred_seq = pred_seq + ["O"] * (gold_len - len(pred_seq))
    return pred_seq[:gold_len]