import nemo.collections.asr.models as nemo_asr_models
from pathlib import Path

model_name = "nvidia/stt_en_fastconformer_ctc_large"
save_dir = Path("models/nemo_asr")

save_dir.mkdir(parents=True, exist_ok=True)

model = nemo_asr_models.EncDecCTCModelBPE.from_pretrained(model_name)
model.save_to(save_dir / f"{model_name}.nemo")

print(f"Model saved to {save_dir / (model_name + '.nemo')}")