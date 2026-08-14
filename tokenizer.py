from __future__ import annotations

from collections.abc import Iterable, Sequence

LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
FEEDBACK_SYMBOLS = ("0", "1", "2")
GUESS_TOKEN = "<G>"
FEEDBACK_TOKEN = "<F>"
END_TOKEN = "<E>"
TOKENS = LETTERS + FEEDBACK_SYMBOLS + (GUESS_TOKEN, FEEDBACK_TOKEN, END_TOKEN)
TOKEN_TO_ID = {token: token_id for token_id, token in enumerate(TOKENS)}
ID_TO_TOKEN = dict(enumerate(TOKENS))
VOCABULARY_SIZE = len(TOKENS)
FEEDBACK_TO_SYMBOL = {"X": "0", "Y": "1", "G": "2"}


def encode(text: str) -> list[int]:
    """Encode Wordle sequence text, treating control markers as single tokens."""
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
            raise ValueError(f"unsupported token: {token!r}") from error
    return token_ids


def decode(token_ids: Iterable[int]) -> str:
    """Decode token IDs back to the exact serialized trajectory text."""
    decoded: list[str] = []
    for token_id in token_ids:
        try:
            decoded.append(ID_TO_TOKEN[token_id])
        except KeyError as error:
            raise ValueError(f"unsupported token ID: {token_id!r}") from error
    return "".join(decoded)


def serialize_trajectory(turns: Sequence[dict[str, object]]) -> str:
    """Serialize ordered guess/feedback turns and terminate with <E>."""
    pieces: list[str] = []
    for turn in turns:
        guess = turn.get("guess")
        feedback = turn.get("feedback")
        if (
            not isinstance(guess, str)
            or len(guess) != 5
            or not guess.isascii()
            or not guess.isalpha()
            or not guess.islower()
        ):
            raise ValueError(f"invalid trajectory guess: {guess!r}")
        if not isinstance(feedback, str) or len(feedback) != 5:
            raise ValueError(f"invalid trajectory feedback: {feedback!r}")
        try:
            encoded_feedback = "".join(
                FEEDBACK_TO_SYMBOL[mark] for mark in feedback
            )
        except KeyError as error:
            raise ValueError(f"invalid feedback mark: {error.args[0]!r}") from error
        pieces.extend((GUESS_TOKEN, guess, FEEDBACK_TOKEN, encoded_feedback))
    pieces.append(END_TOKEN)
    return "".join(pieces)
