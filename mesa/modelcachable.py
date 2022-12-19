"""
A wrapper that extends the model class of mesa by caching functionality.

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
from typing import Any, List


class CacheState(Enum):
    WRITE = 1,
    READ = 2


@staticmethod
def _write_cache_file(cache_file_path: Path, cache_data: List[Any]) -> None:
    """Default function for writing the given cache data to the cache file.
    Used by ModelCachable if not replaced by a custom write function.
    Uses pickle to dump the data into the file.
    Uses gzip for compression."""
    with gzip.open(cache_file_path, 'wb') as file:
        pickle.dump(cache_data, file)


@staticmethod
def _read_cache_file(cache_file_path: Path) -> List[Any]:
    """Default function for reading the cache data from the cache file.
    Used by ModelCachable if not replaced by a custom read function.
    Expects that gzip and pickle have been used to write the file."""
    with gzip.open(cache_file_path, 'rb') as file:
        return pickle.load(file)


@staticmethod
def _write_complete_state_to_json_string(model: Model) -> Any:
    """Default function for writing the current model state into a string.
    Used by ModelCachable if not replaced by a custom write function.
    Uses pickle to dump the complete model.__dict__ to a string"""
    return pickle.dumps(model.__dict__)


@staticmethod
def _load_complete_state_from_json_string(state_json: Any, model: Model) -> None:
    """Default function for reading the current model state from a string.
    Used by ModelCachable if not replaced by a custom read function.
    Expects that the given string is the model.__dict__ dumped by pickle."""
    model.__dict__ = pickle.loads(state_json)


class ModelCachable:
    """Class that takes a model and writes its steps to a cache file."""

    def __init__(self, model: Model, cache_file_path: str, cache_state: CacheState) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
            cache_state: whether to replay by reading from the cache or simulate and write to the cache
        """
        self.model = model
        self.cache_file_path = Path(cache_file_path)
        self._cache_state = cache_state
        self.cache: List[str] = []
        self.step_number: int = 0

        if self._cache_state is CacheState.READ:
            self.read_cache_file()

    def write_state_to_string(self) -> str:
        """Writes the model state to a string.
        Can be overwritten to write just parts of the state or other custom behavior.
        Needs to remain compatible with 'load_state_from_string'.
        """
        return _write_complete_state_to_json_string(self.model)

    def load_state_from_string(self, state_string: str) -> None:
        """Loads the model state from the given string.
        Can be overwritten to load just parts of the state or other custom behavior.
        Needs to remain compatible with 'write_state_to_string'.
        """
        _load_complete_state_from_json_string(state_string, self.model)

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
        print("step")
        if self._cache_state is CacheState.WRITE:
            self.model.step()
            self.cache.append(self.write_state_to_string())

        elif self._cache_state is CacheState.READ:
            model_state_of_step_string = self.cache[self.step_number]
            self.load_state_from_string(model_state_of_step_string)

            # after reading the last step: stop simulation
            if self.step_number == len(self.cache) - 1:
                self.model.running = False

        self.step_number = self.step_number + 1

    def __getattr__(self, item):
        """Act as proxy: forward all attributes (including function calls) from actual model."""
        return self.model.__getattribute__(item)