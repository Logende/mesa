"""
A decorator that wraps the model class of mesa and extends it by caching functionality.

Core Objects: ModelCachable
"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import os
from pathlib import Path
from enum import Enum

import pickle
import gzip
import io
import sys

from mesa import Model

# mypy
from typing import Any, List, Union, IO


class CacheState(Enum):
    WRITE = 1,
    READ = 2


def _write_cache_file(cache_file_path: Path, cache_data: List[Any]) -> None:
    """Default function for writing the given cache data to the cache file.
    Used by ModelCachable if not replaced by a custom write function.
    Uses pickle to dump the data into the file.
    Uses gzip for compression."""
    with gzip.open(cache_file_path, 'wb') as file:
        pickle.dump(cache_data, file)


def _read_cache_file(cache_file_path: Path) -> List[Any]:
    """Default function for reading the cache data from the cache file.
    Used by ModelCachable if not replaced by a custom read function.
    Expects that gzip and pickle have been used to write the file."""
    with gzip.open(cache_file_path, 'rb') as file:
        return pickle.load(file)


def _stream_write_next_chunk_size(stream: IO, size: int):
    chunk_length_bytes = size.to_bytes(length=8, byteorder='little', signed=False)
    stream.write(chunk_length_bytes)


def _stream_read_next_chunk_size(stream):
    return int.from_bytes(stream.read(8), byteorder='little', signed=False)


class ModelCachable:
    """Class that takes a model and writes its steps to a cache file or reads them from a cache file."""

    def __init__(self, model: Model, cache_file_path: Union[str, Path], cache_state: CacheState) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
            cache_state: whether to replay by reading from the cache or simulate and write to the cache
        """
        self.model = model
        self.cache_file_path = Path(cache_file_path)
        self._cache_state = cache_state
        self.cache: List[Any] = []
        self.step_count: int = 0
        self.run_finished = False

        if self._cache_state is CacheState.READ:
            self.read_cache_file()

    def serialize_state(self) -> Any:
        """Serializes the model state.
        Can be overwritten to write just parts of the state or other custom behavior.
        Needs to remain compatible with 'deserialize_state'.
        """
        return pickle.dumps(self.model.__dict__)

    def deserialize_state(self, state: Any) -> None:
        """Deserializes the model state from the given input.
        Can be overwritten to load just parts of the state or other custom behavior.
        Needs to remain compatible with 'serialize_state'.
        """
        self.model.__dict__ = pickle.loads(state)

    def _write_cache_file(self):
        """Writes the cache from memory to 'cache_file_path'.
        Can be overwritten to, for example, use a different file format or compression or destination.
        Needs to remain compatible with 'read_cache_file'
        """
        _write_cache_file(self.cache_file_path, self.cache)
        print("Wrote ModelCachable cache file to " + str(self.cache_file_path))

    def read_cache_file(self):
        """Reads the cache from 'cache_file_path' into memory.
        Can be overwritten to, for example, use a different file format or compression or location.
        Needs to remain compatible with 'write_cache_file'
        """
        self.cache = _read_cache_file(self.cache_file_path)

    def run_model(self) -> None:
        """Run the model until the end condition is reached.
        """
        self.model.run_model()

        self.finish_run()

    def finish_run(self) -> None:
        if self.run_finished:
            raise RuntimeError("ModelCachable: Can not finish a run that was already finished.")

        # model run finished -> write to cache if in writing state
        if self._cache_state is CacheState.WRITE:
            self._write_cache_file()

    def step(self) -> None:
        """A single step."""
        if self._cache_state is CacheState.WRITE:
            self.model.step()
            self.step_write_to_cache()

        elif self._cache_state is CacheState.READ:
            self.step_read_from_cache()

            # after reading the last step: stop simulation
            if self.step_count == len(self.cache) - 1:
                self.model.running = False

        self.step_count = self.step_count + 1

    def step_write_to_cache(self):
        self.cache.append(self.serialize_state())

    def step_read_from_cache(self):
        serialized_state = self.cache[self.step_count]
        self.deserialize_state(serialized_state)

    def __getattr__(self, item):
        """Act as proxy: forward all attributes (including function calls) from actual model."""
        return self.model.__getattribute__(item)


class ModelCachableOptimized(ModelCachable):
    def __init__(self, model: Model, cache_file_path: Union[str, Path], cache_state: CacheState,
                 precision: int = 1, compress_each_step: bool = True):
        super().__init__(model, cache_file_path, cache_state)
        self.precision = precision
        self.compress_each_step = compress_each_step

    def cache_step(self):
        # Cache only every nth step
        if self.step_count % self.precision == 0:
            super().cache_step()

    def compress(self, data: bytes):
        return gzip.compress(data)

    def decompress(self, data: bytes):
        return gzip.decompress(data)

    def serialize_state(self) -> Any:
        """Serializes the model state.
        Can be overwritten to write just parts of the state or other custom behavior.
        Needs to remain compatible with 'deserialize_state'.
        """
        dump = super().serialize_state()
        if self.compress_each_step:
            dump = self.compress(dump)
        return dump

    def deserialize_state(self, state: Any) -> None:
        """Deserializes the model state from the given input.
        Can be overwritten to load just parts of the state or other custom behavior.
        Needs to remain compatible with 'serialize_state'.
        """
        if self.compress_each_step:
            state = self.decompress(state)
        super().deserialize_state(state)


class ModelCachableStreaming(ModelCachableOptimized):

    def __init__(self, model: Model, cache_file_path: Union[str, Path], cache_state: CacheState,
                 precision: int = 1, compress_each_step: bool = True):
        super().__init__(model, cache_file_path, cache_state, precision, compress_each_step)

        if cache_state is CacheState.WRITE:
            if self.cache_file_path.exists():
                print("ModelCachableLarge: cache file (path='" + str(self.cache_file_path) + "') already exists. "
                                                                                             "Deleting it.")
                os.remove(cache_file_path)
            self.cache_file_stream = io.open(cache_file_path, 'wb')

        elif cache_state is CacheState.READ:
            self.cache_file_stream = io.open(cache_file_path, 'rb')

    def finish_run(self) -> None:
        super().finish_run()
        self.cache_file_stream.close()

    def step_write_to_cache(self):
        serialized_state: bytes = self.serialize_state()
        _stream_write_next_chunk_size(self.cache_file_stream, len(serialized_state))
        self.cache_file_stream.write(serialized_state)

    def step_read_from_cache(self):
        chunk_length = _stream_read_next_chunk_size(self.cache_file_stream)
        if chunk_length == 0:
            print("ModelCachableLarge: reached end of cache file stream.")
            self.model.running = False
        else:
            serialized_state = self.cache_file_stream.read(chunk_length)
            self.deserialize_state(serialized_state)

    def _write_cache_file(self):
        if self._cache_state is CacheState.WRITE:
            # end cache file with a chunk size of 0, to make EOF detectable
            _stream_write_next_chunk_size(self.cache_file_stream, 0)

    def read_cache_file(self):
        # nothing to do in advance as a stream is used to read on the go
        return
