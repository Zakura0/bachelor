import os
import json
from openai import OpenAI

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

client = OpenAI()

book_text = open(os.path.join(project_root, "data/raw/verwandlung.txt"), encoding="utf-8").read()
eval_pairs = json.load(open(os.path.join(project_root, "data/processed/eval_pairs_verwandlung.json"), encoding="utf-8"))

hits = 0
total = 0

for pair in eval_pairs[:5]:
    query = pair["summary"]
    expected_spans = [tuple(s) for s in pair["spans"]]

    prompt = f"""Hier ist ein literarischer Text:

{book_text}

Finde die Textstelle, die inhaltlich am besten zu dieser Zusammenfassung passt:
\"{query}\"

Gib nur die Textstelle wortwörtlich zurück, keine Erklärungen."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    answer_clean = answer.strip().strip('"\'„\u201c\u201d\u201f\u00bb\u00ab')

    found_at = book_text.find(answer_clean[:50])
    answer_end = found_at + len(answer_clean) if found_at >= 0 else -1

    hit = False
    if found_at >= 0:
        for expected_start, expected_end in expected_spans:
            if found_at < expected_end and expected_start < answer_end:
                hit = True
                break

    hits += int(hit)
    total += 1

    print(f"Query:    {query[:80]}...")
    print(f"Antwort:  {answer_clean[:100]}...")
    print(f"Position: {found_at}  →  {'HIT' if hit else 'MISS'}")
    print()

print(f"Ergebnis: {hits}/{total} ({hits/total:.1%})")