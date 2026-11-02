"""Gradio web demo.  Run:  python src/app.py"""
import gradio as gr
from vqa_pipeline import ask

EXAMPLES_Q = ["What is in the picture?", "What color is the main object?",
              "How many people are there?", "Is it daytime?"]

def run(image, question, use_blip, use_vilt):
    if image is None:
        return "Please upload an image."
    models = [m for m, on in (("BLIP-VQA", use_blip), ("ViLT-VQA", use_vilt)) if on]
    if not models:
        return "Select at least one model."
    try:
        res = ask(image, question, models)
    except ValueError as e:
        return str(e)
    return "\n".join(f"{r.model}: {r.answer}" + (f"  (conf {r.confidence})" if r.confidence else "")
                     for r in res)

demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil", label="Image"),
            gr.Textbox(label="Question", placeholder=EXAMPLES_Q[0]),
            gr.Checkbox(True, label="BLIP-VQA"), gr.Checkbox(True, label="ViLT-VQA")],
    outputs=gr.Textbox(label="Answers"),
    title="Visual Question Answering",
    description="Upload an image, ask a question, compare two pretrained VQA models.")

if __name__ == "__main__":
    demo.launch()
