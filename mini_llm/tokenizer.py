# The tokenizer: GPT-2's byte-pair encoding (50,257 tokens), through tiktoken.

import tiktoken

_ENC = tiktoken.get_encoding("gpt2")
EOT = _ENC.eot_token  # <|endoftext|> (id 50256): ends every document, and every assistant turn of a chat


def get_tokenizer():
    """The GPT-2 tokenizer. encode_ordinary(text) -> ids (special tokens in the text are not interpreted);
    decode(ids) -> text."""
    return _ENC
