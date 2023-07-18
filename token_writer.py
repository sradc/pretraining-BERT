"""Tools for streaming tokens as bytes into a big file,
and then loading them back into a memory mapped numpy array.
e.g. to tokenize a whole corpus, then access the tokens.
"""
import os
import tempfile
from pathlib import Path
from typing import Union

import numpy as np

DTYPE = "<u2"  # little endian unsigned 16 bit integer. Only have ~40k tokens
# assumes the tokenizer does not produce negative tokens


class TokenWriter:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.exists():
            assert self.path.stat().st_size == 0, f"File {self.path} already exists"
        self.file = open(self.path, "ab")

    def write(self, tokens: np.ndarray):
        assert tokens.dtype == DTYPE
        assert tokens.ndim == 1
        self.file.write(tokens.tobytes())

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_tokens(path: Union[str, Path]) -> np.ndarray:
    # Reading a stream of bytes from a file into a memory mapped array
    file_size = os.path.getsize(path)
    num_elements = file_size // np.dtype(DTYPE).itemsize
    mmapped_array = np.memmap(path, dtype=DTYPE, mode="r", shape=(num_elements,))
    return mmapped_array


if __name__ == "__main__":
    # Informal test
    with tempfile.NamedTemporaryFile() as tmp:
        with TokenWriter(tmp.name) as writer:
            writer.write(np.arange(10, dtype=DTYPE))
            writer.write(np.arange(10, dtype=DTYPE) + 10)
            writer.write(np.arange(10, dtype=DTYPE) + 20)
        # Check can load
        tokens = load_tokens(tmp.name)
        assert np.all(tokens == np.arange(30, dtype=DTYPE))
        print("Successfully streamed tokens to file and loaded them")
        print("tokens:")
        print(tokens)
