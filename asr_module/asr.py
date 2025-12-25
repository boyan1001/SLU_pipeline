import torch
import tempfile
import numpy as np
import wordninja
import os

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import context_biasing


# ============================================================
# Model loading
# ============================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
'''
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path="/datas/store162/kaichen/nemo/models/nemo_asr/nvidia/parakeet-ctc-0.6b.nemo",
    map_location=torch.device(DEVICE)
)
'''
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path="models/nemo_asr/nvidia/stt_en_fastconformer_ctc_large.nemo",
    map_location=torch.device(DEVICE)
)

# asr_model.change_decoding_strategy(decoder_type='ctc')
asr_model.eval()
blank_idx = asr_model.decoding.blank_id


# ============================================================
# Bias list builder (word -> abbr + ninja)
# ============================================================

def build_cb_list_from_words(cb_words):
    tmp_dir = tempfile.mkdtemp()
    cb_list_file = os.path.join(tmp_dir, "cb_list.txt")
    cb_list_file_modified = cb_list_file + ".abbr_and_ninja"

    # step 1: word_word
    with open(cb_list_file, "w", encoding="utf-8") as fn:
        for word in cb_words:
            fn.write(f"{word}_{word}\n")

    # step 2: abbr + wordninja
    with open(cb_list_file, "r", encoding="utf-8") as fn1, \
         open(cb_list_file_modified, "w", encoding="utf-8") as fn2:

        for line in fn1:
            word = line.strip().split("_")[0]
            new_line = f"{word}_{word}"

            # short word -> character split
            if len(word) <= 4 and " " not in word:
                abbr = " ".join(list(word))
                new_line += f"_{abbr}"

            # long word -> wordninja split
            seg = wordninja.split(word)
            if seg and seg[0] != word:
                new_line += f"_{' '.join(seg)}"

            fn2.write(new_line + "\n")

    return cb_list_file_modified


# ============================================================
# ASR inference with optional context biasing
# ============================================================

def run_asr(audio_path, cb_words=None):
    with torch.no_grad():
        hyp = asr_model.transcribe(
            [audio_path],
            batch_size=1,
            return_hypotheses=True
        )[0]

    # 1) 優先嘗試從 hyp 拿 logprobs
    ctc_logprobs = getattr(hyp, "alignments", None)

    if ctc_logprobs is None:
        raise RuntimeError("hyp.alignments is None; need to extract CTC logprobs from model outputs for this NeMo model.")

    ctc_logprobs = ctc_logprobs.cpu().numpy()

    preds = np.argmax(ctc_logprobs, axis=1)
    preds_tensor = torch.tensor(preds).unsqueeze(0)
    '''
    base_text = asr_model.wer.decoding.ctc_decoder_predictions_tensor(
        preds_tensor
    )[0].text
    '''
    base_text = asr_model.decoding.ctc_decoder_predictions_tensor(preds_tensor)[0].text


    if not cb_words:
        return base_text, base_text

    # build context graph
    cb_list_path = build_cb_list_from_words(cb_words)

    context_transcripts = []
    with open(cb_list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("_")
            word = parts[0]
            tokenizations = [
                asr_model.tokenizer.text_to_ids(p)
                for p in parts[1:]
            ]
            context_transcripts.append([word, tokenizations])

    context_graph = context_biasing.ContextGraphCTC(blank_id=blank_idx)
    context_graph.add_to_graph(context_transcripts)

    # run CTC Word Spotter
    ws_results = context_biasing.run_word_spotter(
        ctc_logprobs,
        context_graph,
        asr_model,
        blank_idx=blank_idx,
        beam_threshold=5.0,
        cb_weight=3.0,
        ctc_ali_token_weight=0.6,
    )

    if ws_results:
        bias_text, _ = context_biasing.merge_alignment_with_ws_hyps(
            preds,
            asr_model,
            ws_results,
            decoder_type="ctc",
            blank_idx=blank_idx,
            print_stats=False,
        )
    else:
        bias_text = base_text

    return base_text, bias_text