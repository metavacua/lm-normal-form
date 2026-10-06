# Interface: a readout, version 1

A readout is what a cell produces from a model for the judge.

- **Input:** a model artifact (a Hugging Face repository at a recorded
  revision, possibly converted or quantized by a named tool at a recorded
  revision) and `prepare/prompts.tsv`.
- **Output:** `answers.jsonl` as the judge's input schema says, `model.txt`,
  `model.revision.txt` (the repository's commit at read time), the tool's
  revision, GNU `time` output for each tool run, the server's peak
  resident set from `/proc` when a server was used, and the counts as
  step outputs.
- **Semantics:** for each prompt, the model's next-token distribution over
  its vocabulary after the prompt as given, no chat template, no prefix
  token unless the tokenizer adds one by default (recorded), temperature 0,
  the twenty most probable tokens with their log-probabilities.
- **Two implementations:** llama.cpp's server (`/completion`, `n_probs`
  20) and PyTorch through `transformers` (`softmax(logits[0, -1]).topk(20)`).
  Batch 0003 showed them to agree to four decimals on three models.
- **Compatibility:** changing the number of tokens kept, adding a chat
  template, or changing the decoding of tokens is breaking.
