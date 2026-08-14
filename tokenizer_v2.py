from __future__ import annotations

from collections.abc import Iterable

from tokenizer import TOKENS as V1_TOKENS

MECHANICS_TOKEN = "<M>"
SECRET_TOKEN = "<S>"
POLICY_TOKEN = "<P>"
ROLE_TOKENS = (MECHANICS_TOKEN, SECRET_TOKEN, POLICY_TOKEN)
TOKENS = V1_TOKENS + ROLE_TOKENS
TOKEN_TO_ID = {token: token_id for token_id, token in enumerate(TOKENS)}
ID_TO_TOKEN = dict(enumerate(TOKENS))
VOCABULARY_SIZE = len(TOKENS)


def encode(text: str) -> list[int]:
    """Encode v2 text with task and semantic role markers."""
    token_ids: list[int] = []
    index = 0
    while index < len(text):
        if text[index] == "<":
            marker_end = text.find(">", index + 1)
            if marker_end < 0:
                raise ValueError(f"unterminated control token at character {index}")
            token = text[index : marker_end + 1]
            index = marker_end + 1
        else:
            token = text[index]
            index += 1
        try:
            token_ids.append(TOKEN_TO_ID[token])
        except KeyError as error:
            raise ValueError(f"unsupported v2 token: {token!r}") from error
    return token_ids


def decode(token_ids: Iterable[int]) -> str:
    """Decode v2 token IDs back to their exact serialized text."""
    decoded: list[str] = []
    for token_id in token_ids:
        try:
            decoded.append(ID_TO_TOKEN[token_id])
        except KeyError as error:
            raise ValueError(f"unsupported v2 token ID: {token_id!r}") from error
    return "".join(decoded)
