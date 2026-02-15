import json
from collections import defaultdict

data = json.load(open("data.json"))
summary = data["verwandlung"]["summaries"][0]

summary_to_entries = defaultdict(list)

for anno in summary["annotations"]:
    text_start, text_end = anno["start"], anno["end"]
    text = data["verwandlung"]["text"][text_start:text_end]
    summary_text = " -- ".join([summary["text"][start:end] for start, end in anno["summary_spans"]])
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
        "text": combined_text,
        "summary": summary_text,
        "spans": spans
    })

with open("eval_pairs.json", "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"Created {len(pairs)} text-summary pairs in eval_pairs.json")
print(f"(Merged from {sum(len(e) for e in summary_to_entries.values())} original annotations)")
