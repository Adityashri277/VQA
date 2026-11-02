"""Batch evaluation.  Usage:
    python src/evaluate.py samples/questions.csv
CSV columns: image,question,answer   (image path relative to samples/images/)
Writes outputs/predictions.csv and prints exact-match accuracy per model and per question type.
"""
import sys, time
import pandas as pd
from vqa_pipeline import get_model

def norm(s):
    return str(s).lower().strip().rstrip(".")

def qtype(q):
    q = q.lower()
    if q.startswith(("is", "are", "does", "do", "can")): return "yes/no"
    if q.startswith("how many"): return "number"
    return "other"

def main(path):
    df = pd.read_csv(path)
    rows = []
    for m in ("BLIP-VQA", "ViLT-VQA"):
        model = get_model(m)
        for _, r in df.iterrows():
            t = time.time()
            res = model.answer(f"samples/images/{r.image}", r.question)
            rows.append(dict(model=m, image=r.image, question=r.question, gold=r.answer,
                             pred=res.answer, correct=norm(res.answer) == norm(r.answer),
                             qtype=qtype(r.question), seconds=round(time.time() - t, 2)))
    out = pd.DataFrame(rows)
    out.to_csv("outputs/predictions.csv", index=False)
    print(out.groupby("model")[["correct", "seconds"]].mean().round(3))
    print(out.pivot_table(index="qtype", columns="model", values="correct", aggfunc="mean").round(3))

if __name__ == "__main__":
    main(sys.argv[1])
