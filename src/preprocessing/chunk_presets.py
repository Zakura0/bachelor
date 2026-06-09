# Vordefinierte Chunk-Konfigurationen
# 1 = tiny               (10–30 Wörter,  1 Satz Overlap)
# 2 = small              (10–50 Wörter,  1 Satz Overlap)
# 3 = medium             (30–100 Wörter, 2 Satz Overlap)
# 4 = large              (50–150 Wörter, 2 Satz Overlap)
# 5 = xlarge             (80–200 Wörter, 3 Satz Overlap)
# 6 = medium_high_overlap(30–100 Wörter, 3 Satz Overlap)
# 7 = large_high_overlap (50–150 Wörter, 4 Satz Overlap)
CHUNK_PRESETS = {
    1: dict(name="tiny",                min_words=10,  max_words=30,  overlap=1),
    2: dict(name="small",               min_words=10,  max_words=50,  overlap=1),
    3: dict(name="medium",              min_words=30,  max_words=100, overlap=2),
    4: dict(name="large",               min_words=50,  max_words=150, overlap=2),
    5: dict(name="xlarge",              min_words=80,  max_words=200, overlap=3),
    6: dict(name="medium_high_overlap", min_words=30,  max_words=100, overlap=3),
    7: dict(name="large_high_overlap",  min_words=50,  max_words=150, overlap=4),
}
