import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333"))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4))
box(0.1, 2.2, 1.7, 0.9, "Image\n(upload)", "#dbeafe"); box(0.1, 0.5, 1.7, 0.9, "Question\n(text)", "#dbeafe")
box(2.4, 2.2, 1.9, 0.9, "Image preprocess\nRGB, resize,\nnormalise", "#fef9c3")
box(2.4, 0.5, 1.9, 0.9, "Question cleanup\ntrim, add '?',\ntokenize", "#fef9c3")
box(5.0, 1.35, 2.0, 1.0, "Vision + Text\nencoders\n(ViT / BERT)", "#dcfce7")
box(7.6, 2.2, 1.6, 0.9, "BLIP-VQA\ntext decoder", "#fce7f3"); box(7.6, 0.5, 1.6, 0.9, "ViLT-VQA\nclassifier head", "#fce7f3")
box(9.6, 1.35, 1.3, 1.0, "Answer(s)\n+ confidence\n(Gradio UI)", "#e9d5ff")
arrow(1.9, 2.65, 2.35, 2.65); arrow(1.9, 0.95, 2.35, 0.95)
arrow(4.35, 2.65, 5.0, 2.1); arrow(4.35, 0.95, 5.0, 1.6)
arrow(7.05, 2.1, 7.55, 2.6); arrow(7.05, 1.6, 7.55, 1.0)
arrow(9.25, 2.6, 9.6, 2.0); arrow(9.25, 1.0, 9.6, 1.7)
plt.savefig("docs/pipeline.png", dpi=200, bbox_inches="tight")
