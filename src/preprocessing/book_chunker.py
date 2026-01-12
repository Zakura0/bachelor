from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class SentenceSpan:
    index: int
    start: int
    end: int
    text: str


@dataclass
class Chunk:
    id: int
    start_index: int
    end_index: int
    content: str
    sentence_indices: List[int]
    length: int
    word_count: int


class BookChunker:
    def __init__(
        self,
        min_words: int = 30,
        max_words: int = 120,
        sentence_overlap: int = 1
    ) -> None:
        """
        :param min_words: minimale Wortanzahl pro Chunk
        :param max_words: maximale Wortanzahl pro Chunk
        :param sentence_overlap: Anzahl Sätze, die zwischen Chunks überlappen
        """
        self.min_words = min_words
        self.max_words = max_words
        self.sentence_overlap = max(0, sentence_overlap)

    def build_chunks(self, text: str) -> Dict[str, Any]:
        sentences = self._split_sentences_regex(text)
        chunks = self._build_chunks_from_sentences(text, sentences)

        return {
            "original_text_length": len(text),
            "total_chunks": len(chunks),
            "chunks": [asdict(c) for c in chunks],
        }

    def _split_sentences_regex(self, text: str) -> List[SentenceSpan]:
        """
        Heuristische Satzsegmentierung mit Regex

        Trennung grob nach [.?!…] + Anführungszeichen + Whitespace/Zeilenumbruch.
        """
        pattern = re.compile(
            r".+?(?:(?<=[\.!?…])[\"»“”']*\s+|\Z)",
            re.DOTALL,
        )

        sentences: List[SentenceSpan] = []
        for idx, match in enumerate(pattern.finditer(text)):
            raw_start = match.start()
            raw_end = match.end()
            raw = text[raw_start:raw_end]

            # führende/trailing Whitespaces entfernen, Offsets anpassen
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = raw_start + leading
            end = raw_end - trailing
            sent_text = text[start:end]

            if not sent_text.strip():
                continue

            sentences.append(
                SentenceSpan(
                    index=idx,
                    start=start,
                    end=end,
                    text=sent_text,
                )
            )

        return sentences

    def _build_chunks_from_sentences(
        self,
        full_text: str,
        sentences: List[SentenceSpan],
    ) -> List[Chunk]:
        """
        Baut dynamische Chunks basierend auf einem Wortbudget
        und einer konfigurierbaren Satz-Overlap.
        """
        chunks: List[Chunk] = []
        if not sentences:
            return chunks

        n = len(sentences)
        chunk_id = 1
        i = 0

        while i < n:
            # Starte neuen Chunk bei Satz i
            start_sent_idx = i
            current_sent_idx = i
            current_word_count = 0

            # Füge Sätze hinzu, bis max_words erreicht ist
            while current_sent_idx < n:
                s = sentences[current_sent_idx]
                words_in_s = len(s.text.split())
                # Wenn Chunk noch leer ist, dürfen wir auch einen langen Satz nehmen
                if current_word_count > 0 and current_word_count + words_in_s > self.max_words:
                    break
                current_word_count += words_in_s
                current_sent_idx += 1
                if current_word_count >= self.max_words:
                    break

            while current_word_count < self.min_words and current_sent_idx < n:
                s = sentences[current_sent_idx]
                words_in_s = len(s.text.split())
                current_word_count += words_in_s
                current_sent_idx += 1

            if current_sent_idx <= start_sent_idx:
                break

            window_sentences = sentences[start_sent_idx:current_sent_idx]
            chunk_start = window_sentences[0].start
            chunk_end = window_sentences[-1].end
            content = full_text[chunk_start:chunk_end]

            chunk = Chunk(
                id=chunk_id,
                start_index=chunk_start,
                end_index=chunk_end,
                content=content,
                sentence_indices=[s.index for s in window_sentences],
                length=len(content),
                word_count=current_word_count,
            )
            chunks.append(chunk)
            chunk_id += 1

            # Nächster Chunk startet mit Overlap
            # Beispiel: sentence_overlap=1
            # -> nächster Start = current_sent_idx - 1
            i = current_sent_idx - self.sentence_overlap
            if i <= start_sent_idx:
                i = current_sent_idx  # Schutz vor Endlosschleifen

        return chunks
