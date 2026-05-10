"""
Parse data.json and create eval_pairs_{book}.json for each book (except harrypotter).
Each pair: { "summary": ..., "spans": [(start, end), ...], "text": ... }
Covers all summaries for each book (not just the first one).
"""
import json
import os
from collections import defaultdict

SKIP_BOOKS = {"harrypotter"}

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data.json")

data = json.load(open(data_path, encoding="utf-8"))

total_pairs = 0
total_annos = 0

for book_name, book_data in data.items():
    if book_name in SKIP_BOOKS:
        print(f"Skipping {book_name}")
        continue

    book_text = book_data["text"]
    summaries = book_data["summaries"]

    # Collect all annotations across all summaries
    summary_to_entries = defaultdict(list)

    for summary in summaries:
        for anno in summary["annotations"]:
            text_start, text_end = anno["start"], anno["end"]
            text = book_text[text_start:text_end]
            summary_text = " -- ".join(
                [summary["text"][start:end] for start, end in anno["summary_spans"]]
            )
            summary_to_entries[summary_text].append({
                "text": text,
                "start": text_start,
                "end": text_end
            })

    pairs = []
    for summary_text, entries in summary_to_entries.items():
        combined_text = " [...] ".join([e["text"] for e in entries])
        spans = [(e["start"], e["end"]) for e in entries]
        pairs.append({
            "book": book_name,
            "text": combined_text,
            "summary": summary_text,
            "spans": spans
        })

    out_path = os.path.join(script_dir, f"eval_pairs_{book_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    n_annos = sum(len(e) for e in summary_to_entries.values())
    print(f"{book_name}: {len(pairs)} pairs (merged from {n_annos} annotations) → {out_path}")
    total_pairs += len(pairs)
    total_annos += n_annos

print(f"\nTotal: {total_pairs} pairs across all books (from {total_annos} annotations)")
