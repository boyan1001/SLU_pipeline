import re
import sys

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2

_TAG_PATTERN = re.compile(r'^(O|[BI]-\S+)$')

def is_valid_bio_seq(tags):
    if not isinstance(tags, (list, tuple)):
        return False
    for t in tags:
        if not isinstance(t, str):
            return False
        t = t.strip()
        if not _TAG_PATTERN.match(t):
            return False
    return True

def evaluate_bio_sequences(y_true_all, y_pred_all, verbose: bool = False):
    filtered_true = []
    filtered_pred = []
    skipped = 0

    # chack valid
    for idx, (y_t, y_p) in enumerate(zip(y_true_all, y_pred_all)):
        if not is_valid_bio_seq(y_t) or not is_valid_bio_seq(y_p):
            skipped += 1
            if verbose:
                print(f"[slot_metrics] skip sample #{idx}: invalid BIO tag found",
                      file=sys.stderr)
            continue

        if len(y_t) != len(y_p):
            skipped += 1
            if verbose:
                print(f"[slot_metrics] skip sample #{idx}: length mismatch "
                      f"(true={len(y_t)}, pred={len(y_p)})",
                      file=sys.stderr)
            continue

        filtered_true.append(y_t)
        filtered_pred.append(y_p)

    if len(filtered_true) == 0:
        if verbose:
            print("[slot_metrics] no valid samples left; return 0 metrics "
                  f"(skipped={skipped})",
                  file=sys.stderr)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "invalid": skipped}

    # entity-level evaluation
    prec = precision_score(filtered_true, filtered_pred, mode='strict', scheme=IOB2)
    rec  = recall_score(filtered_true, filtered_pred, mode='strict', scheme=IOB2)
    f1   = f1_score(filtered_true, filtered_pred, mode='strict', scheme=IOB2)
    return {"precision": prec, "recall": rec, "f1": f1, "invalid": skipped}

def get_metrics(pred_intent, real_intent, pred_slots, real_slots, verbose: bool = False):
    num_samples = len(real_intent)

    # === Intent acc ===
    intent_correct = 0

    for p_intent, r_intent in zip(pred_intent, real_intent):
        if p_intent == r_intent:
            intent_correct += 1
    
    intent_acc = intent_correct / num_samples if num_samples > 0 else 0.0

    # === Overal accuracy ===
    joint_total = 0
    joint_correct = 0

    for idx, (p_intent, r_intent, p_slots, r_slots) in enumerate(zip(pred_intent, real_intent, pred_slots, real_slots)):
        if not is_valid_bio_seq(r_slots) or not is_valid_bio_seq(p_slots):
            if verbose:
                print(f"[overall_acc] skip sample #{idx}: invalid BIO tag found",
                      file=sys.stderr)
            continue

        if len(r_slots) != len(p_slots):
            if verbose:
                print(f"[overall_acc] skip sample #{idx}: length mismatch "
                      f"(true={len(r_slots)}, pred={len(p_slots)})",
                      file=sys.stderr)
            continue

        joint_total += 1

        if (p_intent == r_intent) and (p_slots == r_slots):
            joint_correct += 1

    overall_acc = joint_correct / joint_total if joint_total > 0 else 0.0

    # === Slot Metrics ===
    slot_metrics = evaluate_bio_sequences(real_slots, pred_slots, verbose)
    slot_prec = slot_metrics["precision"]
    slot_rec = slot_metrics["recall"]
    slot_f1 = slot_metrics["f1"]
    invalid = slot_metrics["invalid"]

    args = {
        "intent_acc": intent_acc,
        "slot_prec": slot_prec,
        "slot_rec": slot_rec,
        "slot_f1": slot_f1,
        "overall_acc": overall_acc,
        "invalid": invalid,
    }

    return args