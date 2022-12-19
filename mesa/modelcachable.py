"""
A wrapper that extends the model class of Mesa by caching functionality.

Core Objects: ModelCachable
"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import os.path as path
from enum import Enum

import json

from mesa import Model

# mypy
from typing import List


class CacheState(Enum):
    WRITING = 1,
    READING = 2


@staticmethod
def _write_cache_file(cache_file_path: path, cache_data: List[str]) -> None:
    json.dump(cache_data, cache_file_path)
    # todo: add file compression


@staticmethod
def _read_cache_file(cache_file_path: path) -> List[str]:
    return json.load(cache_file_path)


@staticmethod
def _write_complete_state_to_json_string(model: Model) -> str:
    model_dict_copy = model.__dict__.copy()
    # remove random because it is not JSON serializable
    model_dict_copy["random"] = None
    return json.dumps(model_dict_copy)


@staticmethod
def _load_complete_state_from_json_string(state_json_string: str, model: Model) -> None:
    existing_random = model.random
    state_dict = json.loads(state_json_string)
    model.__dict__ = state_dict
    model.random = existing_random


class ModelCachable:
    """Class that takes a model and writes its steps to a cache file."""

    def __init__(self, model: Model, cache_file_path: path, cache_state: CacheState) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
        """
        self.model = model
        self.cache_file_path = cache_file_path
        self._cache_state = cache_state
        self.cache: List[str] = []
        self.step_number: int = 0

        if self._cache_state is CacheState.READING:
            self.read_cache_file()

    def write_state_to_string(self) -> str:
        """Writes the model state to a string. Needs to be compatible with 'load_state_from_string'.
        Can be overwritten to write just parts of the state or other custom behavior.
        """
        return _write_complete_state_to_json_string(self.model)

    def load_state_from_string(self, state_string: str) -> None:
        """Loads the model state from the given string. Needs to be compatible with 'write_state_to_string'.
        Can be overwritten to load just parts of the state or other custom behavior.
        """
        _load_complete_state_from_json_string(state_string, self.model)

    def write_cache_file(self):
        """Writes the cache from memory to 'cache_file_path'.
        Can be overwritten to, for example, use a different file format or compression or destination.
        Needs to remain compatible with 'read_cache_file'
        """
        _write_cache_file(self.cache_file_path, self.cache)

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
        if self._cache_state is CacheState.WRITING:
            self.write_cache_file()

    def step(self) -> None:
        """A single step."""
        if self._cache_state is CacheState.WRITING:
            self.model.step()
            self.cache.append(self.write_state_to_string())

        elif self._cache_state is CacheState.READING:
            model_state_of_step_string = self.cache[self.step_number]
            self.load_state_from_string(model_state_of_step_string)

            # after reading the last step: stop simulation
            if self.step_number == len(self.cache) - 1:
                self.model.running = False

        self.step_number = self.step_number + 1

    def next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        return self.model.next_id()

    def reset_randomizer(self, seed: int | None = None) -> None:
        """Reset the model random number generator.

        Args:
            seed: A new seed for the RNG; if None, reset using the current seed
        """
        self.model.reset_randomizer()

    def initialize_data_collector(
        self, model_reporters=None, agent_reporters=None, tables=None
    ) -> None:
        self.model.initialize_data_collector(model_reporters, agent_reporters, tables)
