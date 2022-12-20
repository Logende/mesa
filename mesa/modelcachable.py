"""
A decorator that wraps the model class of mesa and extends it by caching functionality.

Core Objects: ModelCachable
"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

from pathlib import Path
from enum import Enum

import pickle
import gzip

from mesa import Model

# mypy
from typing import Any, List, Union


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

    def write_cache_file(self):
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

        # model run finished -> write to cache if in writing state
        if self._cache_state is CacheState.WRITE:
            self.write_cache_file()

    def step(self) -> None:
        """A single step."""
        if self._cache_state is CacheState.WRITE:
            self.model.step()
            self.cache_step()

        elif self._cache_state is CacheState.READ:
            model_state_of_step_string = self.cache[self.step_count]
            self.deserialize_state(model_state_of_step_string)

            # after reading the last step: stop simulation
            if self.step_count == len(self.cache) - 1:
                self.model.running = False

        self.step_count = self.step_count + 1

    def cache_step(self):
        self.cache.append(self.serialize_state())

    def __getattr__(self, item):
        """Act as proxy: forward all attributes (including function calls) from actual model."""
        return self.model.__getattribute__(item)


class ModelCachableLarge(ModelCachable):

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

    def write_cache_file(self):
        """Writes the cache from memory to 'cache_file_path'.
        Can be overwritten to, for example, use a different file format or compression or destination.
        Needs to remain compatible with 'read_cache_file'
        """
        _write_cache_file(self.cache_file_path, self.cache)
        print("Wrote ModelCachable cache file to " + str(self.cache_file_path))
        # TODO: write to filestream / append on existing file with every step instead of writing everything here

    def read_cache_file(self):
        """Reads the cache from 'cache_file_path' into memory.
        Can be overwritten to, for example, use a different file format or compression or location.
        Needs to remain compatible with 'write_cache_file'
        """
        self.cache = _read_cache_file(self.cache_file_path)