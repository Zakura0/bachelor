import json
import re
import os

def chunk(text):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    chunks = []
    current_pos = 0
    
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            start_pos = text.find(paragraph.strip(), current_pos)
            end_pos = start_pos + len(paragraph.strip())
            
            chunk = {
                "id": i+1,
                "content": paragraph.strip(),
                "start_index": start_pos,
                "end_index": end_pos,
                "length": len(paragraph.strip()),
                "word_count": len(paragraph.strip().split())
            }
            
            chunks.append(chunk)
            current_pos = end_pos
    
    return chunks

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'data', 'raw', 'verwandlung.txt')
    output_path = os.path.join(project_root, 'data', 'processed', 'verwandlung_absaetze.json')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunk(text)
    
    result = {
        "original_text_length": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"{len(chunks)} Chunks erstellt und in {output_path} gespeichert")

if __name__ == "__main__":
    main()
