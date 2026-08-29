# Attribution and modification notice: Yambda-50M likes subset

- Source dataset: Yambda-5B / Yambda-50M likes
- Publisher: Yandex LLC
- Dataset card: https://huggingface.co/datasets/yandex/yambda
- Source revision: `dd6f3a19eef5866e346c3270e098baa641a44948`
- Source file: `flat/50m/likes.parquet`, verified by SHA-256 in
  `data/datasets.json`
- License: Apache License 2.0

ML Mentor created a deterministic educational subset. From users with at
least 20 likes, it selects the first 2,000 users ordered by SHA-256 of
`ml-mentor-recsys-v1:<uid>`. It then retains items with at least two likes in
that group, keeps only `uid`, `item_id`, `timestamp`, and `is_organic`, and
sorts rows by user, timestamp, and item. Values are not otherwise modified.
