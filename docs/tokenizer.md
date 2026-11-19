# Tokenizer

Byte-pair encoding trained on the sports corpus: 2048 tokens (256 bytes, 1791 merges and one end-of-text token).

- Corpus: 4,246,213 characters become 1,377,928 tokens, **3.08 characters per token** (one token per character would be 1.00).
- Round trip is exact: decode(encode(corpus)) == corpus is True.

Tokenizing a sentence (`|` separates tokens):

    M|anchester| United| beat| Real| Madrid| 2|-|1| in| the| Champions| League| final| on| Saturday|.

Longest learned tokens: ' international', ' International', ' Championships', ' Championship', ' championship', ' competition', ' Paralympics', ' performance'
