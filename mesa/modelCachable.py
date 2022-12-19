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
from mesa.datacollection import DataCollector

# mypy
from typing import Any, List


class CacheState(Enum):
    INACTIVE = 1,
    WRITING = 2,
    READING = 3


@staticmethod
def _write_cache_file(cache_file_path: path, cache_data: List[str]) -> None:
    print("TODO")


@staticmethod
def _read_cache_file(cache_file_path: path) -> List[str]:
    return ["TODO"]


@staticmethod
def _write_complete_state_to_json_string(model: Model) -> str:
    return json.dumps(model.__dict__)


@staticmethod
def _load_complete_state_from_json_string(state_json_string: str, model: Model) -> None:
    state_dict = json.loads(state_json_string)
    model.__dict__ = state_dict


class ModelCachable:
    """Class that takes a model and writes its steps to a cache file."""

    def __init__(self, model: Model, cache_file_path: path, cache_state: CacheState = CacheState.INACTIVE) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
        """
        self.model = model
        self.cache_file_path = cache_file_path
        self.cache_state = cache_state
        self.cache: List[str] = []
        self.step_number: int = 0

    def _set_cache_state(self, cache_state: CacheState) -> None:
        # When no state change: do nothing
        if self.cache_state == cache_state:
            print("ModelCachable: requested new cache_state " + str(cache_state) + "but already was in that state. "
                                                                                   "Doing nothing.")
            return

        # State change and previously was writing: save cache to file
        if self.cache_state == CacheState.WRITING:
            self.write_cache_file()

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
        Can be overwritten to, for example, use a different file format or compression or destination.
        Needs to remain compatible with 'read_cache_file'
        """
        self.cache = _read_cache_file(self.cache_file_path)

    def run_model(self) -> None:
        """Run the model until the end condition is reached.
        """
        self.model.run_model()

    def step(self) -> None:
        """A single step."""
        if self.cache_state is CacheState.INACTIVE:
            self.model.step()

        elif self.cache_state is CacheState.WRITING:
            self.model.step()
            self.cache.append(self.write_state_to_string())

        elif self.cache_state is CacheState.READING:
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
