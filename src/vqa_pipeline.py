"""Core Visual Question Answering pipeline.

Two pretrained models are wrapped behind one interface:
  * BLIP-VQA  : Salesforce/blip-vqa-base            (generative answer decoder)
  * ViLT-VQA  : dandelin/vilt-b32-finetuned-vqa     (classification over 3129 answers)
"""
import re
from dataclasses import dataclass
import torch
from PIL import Image
from transformers import (BlipProcessor, BlipForQuestionAnswering,
                          ViltProcessor, ViltForQuestionAnswering)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(src) -> Image.Image:
    """Accept a path or PIL image and return an RGB PIL image."""
    img = Image.open(src) if isinstance(src, str) else src
    return img.convert("RGB")


def clean_question(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip())
    if not q:
        raise ValueError("Question is empty.")
    return q if q.endswith("?") else q + "?"


@dataclass
class VQAResult:
    model: str
    answer: str
    confidence: float | None = None


class BlipVQA:
    name = "BLIP-VQA"
    def __init__(self, ckpt="Salesforce/blip-vqa-base"):
        self.proc = BlipProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, image, question) -> VQAResult:
        inputs = self.proc(load_image(image), clean_question(question),
                           return_tensors="pt").to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=20)
        return VQAResult(self.name, self.proc.decode(out[0], skip_special_tokens=True))


class ViltVQA:
    name = "ViLT-VQA"
    def __init__(self, ckpt="dandelin/vilt-b32-finetuned-vqa"):
        self.proc = ViltProcessor.from_pretrained(ckpt)
        self.model = ViltForQuestionAnswering.from_pretrained(ckpt).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, image, question) -> VQAResult:
        inputs = self.proc(load_image(image), clean_question(question),
                           return_tensors="pt").to(DEVICE)
        logits = self.model(**inputs).logits
        idx = logits.argmax(-1).item()
        conf = torch.sigmoid(logits)[0, idx].item()
        return VQAResult(self.name, self.model.config.id2label[idx], round(conf, 3))


_CACHE = {}
def get_model(name: str):
    if name not in _CACHE:
        _CACHE[name] = {"BLIP-VQA": BlipVQA, "ViLT-VQA": ViltVQA}[name]()
    return _CACHE[name]


def ask(image, question, models=("BLIP-VQA", "ViLT-VQA")):
    return [get_model(m).answer(image, question) for m in models]
