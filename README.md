Bachelor Thesis Project

# Retrieval Pipeline Overview

This project implements a multi-stage retrieval pipeline for searching and ranking text chunks from literary works. Below are the main steps of the process:

## 1. Chunking
- The input text is split into sentences using a regex-based heuristic.
- Sentences can be grouped into overlapping chunks based on word count constraints.
- Each chunk is stored with metadata (start/end position, sentence indices, word count).

## 2. Sparse Retrieval (TF-IDF)
- All chunks are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
- A user query is also vectorized.
- Chunks are ranked by cosine similarity to the query vector.

## 3. Dense Retrieval (Embeddings)
- Chunks and queries are encoded into semantic embeddings using a SentenceTransformer model.
- Chunks are ranked by similarity (dot product/cosine similarity) to the query embedding.

## 4. Reciprocal Rank Fusion (RRF)
- The top results from TF-IDF and Embedding retrieval are merged using Reciprocal Rank Fusion.
- This produces a single candidate list, favoring chunks ranked highly by both methods.

## 5. Cross-Encoder Reranking
- The best candidates are reranked using a CrossEncoder model.
- Each (query, chunk) pair is scored for relevance by jointly encoding both texts.
- The top results are selected for final output.

## 6. Natural Language Inference (NLI)
- The top reranked chunks are further evaluated using an NLI model.
- For each chunk, the model predicts whether it entails, contradicts, or is neutral to the query.
- Chunks with high entailment scores are prioritized in the final results.
