"""
A decorator that wraps the model class of mesa and extends it by caching functionality.

Core Objects: ModelCachable
"""

import os
import io
from pathlib import Path
from enum import Enum
from typing import Any, List, Union, IO

import dill
import gzip

from mesa import Model


class CacheState(Enum):
    WRITE = 1,
    READ = 2


def _write_cache_file(cache_file_path: Path, cache_data: List[Any]) -> None:
    """Default function for writing the given cache data to the cache file.
    Used by ModelCachable if not replaced by a custom write function.
    Uses dill to dump the data into the file.
    Uses gzip for compression."""
    with gzip.open(cache_file_path, 'wb') as file:
        dill.dump(cache_data, file)


def _read_cache_file(cache_file_path: Path) -> List[Any]:
    """Default function for reading the cache data from the cache file.
    Used by ModelCachable if not replaced by a custom read function.
    Expects that gzip and dill have been used to write the file."""
    with gzip.open(cache_file_path, 'rb') as file:
        return dill.load(file)


def _stream_write_next_chunk_size(stream: IO, size: int):
    chunk_length_bytes = size.to_bytes(length=8, byteorder='little', signed=False)
    stream.write(chunk_length_bytes)


def _stream_read_next_chunk_size(stream):
    return int.from_bytes(stream.read(8), byteorder='little', signed=False)


class ModelCachable:
    """Class that takes a model and writes its steps to a cache file or reads them from a cache file."""

    def __init__(self, model: Model, cache_file_path: Union[str, Path], cache_state: CacheState,
                 cache_step_rate: int = 1) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
            cache_state: whether to replay by reading from the cache or simulate and write to the cache
            cache_step_rate: only every n-th step is cached. If it is 1, every step is cached. If it is 2,
            only every second step is cached and so on. Increasing 'cache_step_rate' will reduce cache size and
            increase replay performance by skipping the steps inbetween every n-th step.
        """
        self.model = model
        self.cache_file_path = Path(cache_file_path)
        self._cache_state = cache_state
        self._cache_step_rate = cache_step_rate

        self.cache: List[Any] = []
        self.step_count: int = 0
        self.run_finished = False

        if self._cache_state is CacheState.READ:
            self._read_cache_file()

    def _serialize_state(self) -> Any:
        """Serializes the model state.
        Can be overwritten to write just parts of the state or other custom behavior.
        Needs to remain compatible with '_deserialize_state'.

        Note that for large model states, it might make sense to add compression during the serialization.
        That way the size of the cache in memory can be reduced. Additionally, while, by default, the resulting output
        cache file is compressed too (see '_write_cache_file'), this is not the case, when using other file handling
        behavior, such as writing to a buffered file stream during every step (see 'ModelCachableStreaming'). For such
        use-cases, a way to reduce the size of the resulting output cache file is to compress the individual steps. That
        way, for example, reading the cache from the file stream step by step remains possible, without having to
        load the complete cache into memory. This is not possible, when the complete output file is compressed.
        """
        return dill.dumps(self.model.__dict__)

    def _deserialize_state(self, state: Any) -> None:
        """Deserializes the model state from the given input.
        Can be overwritten to load just parts of the state, decompress data, or other custom behavior.
        Needs to remain compatible with '_serialize_state'.
        """
        self.model.__dict__ = dill.loads(state)

    def _write_cache_file(self) -> None:
        """Writes the cache from memory to 'cache_file_path'.
        Can be overwritten to, for example, use a different file format or compression or destination.
        Needs to remain compatible with '_read_cache_file'.
        """
        _write_cache_file(self.cache_file_path, self.cache)
        print("Wrote ModelCachable cache file to " + str(self.cache_file_path))

    def _read_cache_file(self) -> None:
        """Reads the cache from 'cache_file_path' into memory.
        Can be overwritten to, for example, use a different file format or compression or location.
        Needs to remain compatible with '_write_cache_file'
        """
        self.cache = _read_cache_file(self.cache_file_path)

    def run_model(self) -> None:
        """Run the model until the end condition is reached.
        """
        # self.model.run_model()
        # Right now if someone has a custom run_model function, they need to overwrite this function too

        while self.model.running:
            self.step()

        self.finish_run()

    def finish_run(self) -> None:
        """Tells the caching functionality that the run is finished and operations such as writing the cache
        file can be performed. Automatically called by the 'run_model' function after the run, but needs to be
        manually called, when calling the steps manually."""
        if self.run_finished:
            print("ModelCachable: tried to finish run that was already finished. Doing nothing.")
            return

        # model run finished -> write to cache if in writing state
        if self._cache_state is CacheState.WRITE:
            self._write_cache_file()

        self.run_finished = True

    def step(self) -> None:
        """A single step."""
        if self._cache_state is CacheState.WRITE:
            self.model.step()
            # Cache only every n-th step
            if (self.step_count + 1) % self._cache_step_rate == 0:
                self._step_write_to_cache()

        elif self._cache_state is CacheState.READ:
            self._step_read_from_cache()

            # after reading the last step: stop simulation
            if self.step_count == len(self.cache) - 1:
                self.model.running = False

        self.step_count = self.step_count + 1

    def _step_write_to_cache(self) -> None:
        """Is performed for every step, when 'cache_state' is 'WRITE'. Serializes the current state of the model and
        adds it to the cache (which is a list that contains the state for each performed step)."""
        self.cache.append(self._serialize_state())

    def _step_read_from_cache(self) -> None:
        """Is performed for every step, when 'cache_state' is 'READ'. Reads the next state from the cache, deserializes
        it and then updates the model state to this new state."""
        serialized_state = self.cache[self.step_count]
        self._deserialize_state(serialized_state)

    def __getattr__(self, item):
        """Act as proxy: forward all attributes (including function calls) from actual model."""
        return self.model.__getattribute__(item)


class ModelCachableStreaming(ModelCachable):
    """Decorator for ModelCachableOptimized that uses buffered streams for reading and writing the cache, instead
    of keeping the complete cache in memory. Useful when the cache is large."""

    def __init__(self, model: Model, cache_file_path: Union[str, Path], cache_state: CacheState,
                 cache_step_rate: int = 1):
        super().__init__(model, cache_file_path, cache_state, cache_step_rate)

        if cache_state is CacheState.WRITE:
            if self.cache_file_path.exists():
                print("ModelCachableLarge: cache file (path='" + str(self.cache_file_path) + "') already exists. "
                                                                                             "Deleting it.")
                os.remove(cache_file_path)
            self.cache_file_stream = io.open(cache_file_path, 'wb')

        elif cache_state is CacheState.READ:
            self.cache_file_stream = io.open(cache_file_path, 'rb')

    def finish_run(self) -> None:
        """Tells the caching functionality that the run is finished and operations such as writing the cache
        file can be performed. Automatically called by the 'run_model' function after the run, but needs to be
        manually called, when calling the steps manually."""
        super().finish_run()
        self.cache_file_stream.close()

    def _step_write_to_cache(self) -> None:
        """Is performed for every step, when 'cache_state' is 'WRITE'. Serializes the current state of the model and
        writes it to the cache file stream."""
        serialized_state: bytes = self._serialize_state()
        _stream_write_next_chunk_size(self.cache_file_stream, len(serialized_state))
        self.cache_file_stream.write(serialized_state)

    def _step_read_from_cache(self) -> None:
        """Is performed for every step, when 'cache_state' is 'READ'. Reads the next state from the cache file stream,
        deserializes it and then updates the model state to this new state."""
        chunk_length = _stream_read_next_chunk_size(self.cache_file_stream)
        if chunk_length == 0:
            print("ModelCachableLarge: reached end of cache file stream.")
            self.model.running = False
        else:
            serialized_state = self.cache_file_stream.read(chunk_length)
            self._deserialize_state(serialized_state)

    def _write_cache_file(self) -> None:
        """Overwrites the '_write_cache_file' function of the ModelCachable class. As the file content is written
        to the stream during each step, this function does not have to write the complete cache file.
        It only adds an EOF hint to the cache file stream. After that, the stream can be closed and the cache file is
        completed.
        """
        # end cache file with a chunk size of 0, to make EOF detectable
        _stream_write_next_chunk_size(self.cache_file_stream, 0)

    def _read_cache_file(self) -> None:
        """Overwrites the '_read_cache_file' function of the ModelCachable class. As the file content is read from
        the stream during each step, this function does not have to do anything in advance."""
        return
