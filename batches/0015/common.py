# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Text for the real-model scripts of batch 0015: eight plain prompts, and windows of the WikiText-2 test set (the same as batch 0015's, which this batch does not depend on).
PLAIN_PROMPTS = ["The capital of France is", "Water boils at one hundred degrees Celsius at sea level, and", "In 1969, the first human to walk on the Moon was",
                 "The three primary colors are red, blue, and", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "Once upon a time, in a small village by the sea,",
                 "The derivative of x squared with respect to x is", "To make a cup of tea, first"]


def wikitext_windows(tok, n=16, size=512):
    """n windows of `size` token ids from the WikiText-2 raw test set, and the text they were cut from."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    ids = tok(text)["input_ids"]
    wins = [ids[i * size:(i + 1) * size] for i in range(n) if (i + 1) * size <= len(ids)]
    return wins, tok.decode(sum(wins, []))
