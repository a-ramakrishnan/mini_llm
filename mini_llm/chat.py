# The chat format: conversations as plain text that GPT-2's tokenizer handles without new tokens.
#
#   <|endoftext|>System: Be brief.                 (only if the conversation has one)
#
#   User: What is the water cycle?
#
#   Assistant: The water cycle is ...<|endoftext|>User: ...
#
# Every assistant turn ends with <|endoftext|>, so generation stops there. During instruction tuning the loss counts
# only the assistant's tokens (and that <|endoftext|>): the model learns to answer, not to write questions.

CHAT_FORMAT = "smoltalk-plain-v1"
ROLE_PREFIX = {"system": "System: ", "user": "User: ", "assistant": "Assistant: "}


def encode_conversation(messages, enc):
    """(ids, mask) for a list of {"role", "content"} messages. mask is 1 on the tokens the model should learn to
    write: the assistant's answers and the <|endoftext|> that ends each. A last message with content None is an open
    assistant turn: the ids stop after "Assistant:"."""
    eot = enc.eot_token
    ids, mask = [eot], [0]
    for k, m in enumerate(messages):
        role = m["role"] if m["role"] in ROLE_PREFIX else "user"
        sep = "" if k == 0 or messages[k - 1]["role"] == "assistant" else "\n\n"   # after <|endoftext|>: nothing
        head = enc.encode_ordinary(sep + ROLE_PREFIX[role].rstrip())       # "Assistant:"; the space starts the body
        ids += head
        mask += [0] * len(head)
        if m["content"] is None:            # an open assistant turn: the prompt ends here
            break
        body = enc.encode_ordinary(" " + m["content"].strip())
        if role == "assistant":
            ids += body + [eot]
            mask += [1] * (len(body) + 1)
        else:
            ids += body
            mask += [0] * len(body)
    return ids, mask


def format_prompt(messages, enc):
    """Token ids for a conversation so far, ending with "Assistant:" so the model writes the next answer."""
    ids, _ = encode_conversation(list(messages) + [{"role": "assistant", "content": None}], enc)
    return ids


def fit_to_context(messages, enc, block_size, reserve=160):
    """Drop the oldest turns (keeping a leading system message) until the prompt leaves `reserve` tokens for the
    answer inside the model's context. Returns (messages kept, prompt ids)."""
    messages = list(messages)
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(head):]
    while True:
        ids = format_prompt(head + rest, enc)
        if len(ids) + reserve <= block_size or len(rest) <= 1:
            return head + rest, ids[-(block_size - 1):]
        rest = rest[2:] if len(rest) > 2 else rest[1:]   # a question and its answer go together
