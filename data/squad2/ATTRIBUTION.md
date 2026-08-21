# Attribution and modification notice: SQuAD 2.0 subset

- Source dataset: Stanford Question Answering Dataset 2.0 (SQuAD 2.0)
- Authors: Pranav Rajpurkar, Robin Jia, Percy Liang and collaborators
- Project: https://rajpurkar.github.io/SQuAD-explorer/
- Source file: `dev-v2.0.json`, verified by SHA-256 in `data/datasets.json`
- License: CC BY-SA 4.0

ML Mentor created a deterministic educational subset. It keeps the first 100
eligible contexts in source order, one answerable question per context, and
one impossible question for the first 50 eligible contexts that contain one.
Duplicate answer spans are removed and internal field names are normalized.
Question, context and answer text are otherwise unchanged. This file is an
adaptation, so the CC BY-SA 4.0 share-alike requirement continues to apply.
