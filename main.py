import gradio as gr
import tempfile
import soundfile as sf

import asr_module.asr as asr
import nlu_module.nlu as nlu

# Gradio callback
def infer(audio, bias_text):
    if audio is None:
        return "", ""

    sr, wav = audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
        sf.write(audio_path, wav, sr)

    cb_words = None
    if bias_text.strip():
        cb_words = [
            w.strip().lower()
            for w in bias_text.splitlines()
            if w.strip()
        ]

    no_bias, with_bias = asr.run_asr(audio_path, cb_words)
    return no_bias, with_bias

# total pipeline
def pipeline(audio, bias_text, dataset, model, stage):
    no_bias, with_bias = infer(audio, bias_text)
    base_intent, base_slot, bias_intent, bias_slot = nlu.nlu_pipeline(
        no_bias,
        with_bias,
        dataset,
        model,
        stage
    )
    return no_bias, with_bias, base_intent, base_slot, bias_intent, bias_slot


def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("ASR"):
                gr.Markdown("## Parakeet CTC Context Biasing Demo")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Input Audio (wav)"
                )

                bias_input = gr.Textbox(
                    label="Bias Words (one word per line)",
                    lines=6,
                    placeholder=""
                )

                run_btn = gr.Button("Run ASR")
                
                out_no_bias = gr.Textbox(label="ASR Result (No Bias)")
                out_bias = gr.Textbox(label="ASR Result (With Bias)")

                run_btn.click(
                    infer,
                    inputs=[audio_input, bias_input],
                    outputs=[out_no_bias, out_bias]
                )
            with gr.TabItem("NLU"):
                gr.Markdown("## GPT-SLU Intect Detection and Slot Filling Demo")

                utterance_input = gr.Textbox(
                    label="Input Utterance",
                    lines=2,
                    placeholder="Type your utterance here..."
                )

                model_input = gr.Radio(
                    label="Select Model",
                    choices=["gemma3:4b", "gemma3:12b", "gpt-oss:20b"],
                    value="gemma3:4b"
                )

                dataset_input = gr.Radio(
                    label="Select Dataset",
                    choices=["SNIPS", "ATIS", "SLURP"],
                    value="SLURP"
                )

                stage_input = gr.Radio(
                    label="Select Number of Stages",
                    choices=[1, 2, 3],
                    value=2
                )

                run_btn = gr.Button("Run NLU")

                out_intent = gr.Textbox(label="Predicted Intent")
                out_slots = gr.Textbox(label="Predicted Slots")

                bias_input = gr.State(value="")
                run_btn.click(
                    nlu.nlu_pipeline,
                    inputs=[
                        utterance_input,
                        bias_input,
                        dataset_input,
                        model_input,
                        stage_input
                    ],
                    outputs=[out_intent, out_slots]
                )

            with gr.TabItem("ASR + NLU"):
                gr.Markdown("## SLU Pipeline Demo")
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Input Audio (wav)"
                        )

                        bias_input = gr.Textbox(
                            label="Bias Words (one word per line)",
                            lines=6,
                            placeholder=""
                        )

                        model_input = gr.Radio(
                            label="Select Model",
                            choices=["gemma3:4b", "gemma3:12b", "gpt-oss:20b"],
                            value="gemma3:4b"
                        )

                        dataset_input = gr.Radio(
                            label="Select Dataset",
                            choices=["SNIPS", "ATIS", "SLURP"],
                            value="SLURP"
                        )

                        stage_input = gr.Radio(
                            label="Select Number of Stages",
                            choices=[1, 2, 3],
                            value=2
                        )

                        run_btn = gr.Button("Run SLU Pipeline")

                    with gr.Column():
                        out_no_bias = gr.Textbox(label="ASR Result (No Bias)")
                        out_bias = gr.Textbox(label="ASR Result (With Bias)")
                        
                        with gr.Group():
                            gr.Markdown("### No Bias Prediction")
                            out_intent = gr.Textbox(label="Predicted Intent")
                            out_slots = gr.Textbox(label="Predicted Slots")

                        with gr.Group():
                            gr.Markdown("### With Bias Prediction")
                            out_bias_intent = gr.Textbox(label="Predicted Intent (With Bias)")
                            out_bias_slots = gr.Textbox(label="Predicted Slots (With Bias)")

                run_btn.click(
                    pipeline,
                    inputs=[
                        audio_input,
                        bias_input,
                        dataset_input,
                        model_input,
                        stage_input
                    ],
                    outputs=[
                        out_no_bias,
                        out_bias,
                        out_intent,
                        out_slots,
                        out_bias_intent,
                        out_bias_slots
                    ]
                )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
