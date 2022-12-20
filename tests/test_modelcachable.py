import gzip
import pickle
import lzma
import dill

from mesa.modelcachable import Model, ModelCachable, CacheState, ModelCachableOptimized, ModelCachableStreaming, \
    _stream_read_next_chunk_size

from unittest.mock import MagicMock
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path


class ModelFibonacci(Model):
    """Simple fibonacci model to be used by the tests."""
    previous = 0
    current = 1

    def step(self):
        new_value = self.previous + self.current
        self.previous = self.current
        self.current = new_value
        if new_value > 100000:
            self.running = False

    def custom_model_function(self):
        return self.current


class ModelFibonacciForReplay(ModelFibonacci):
    """Same as the fibonacci model, except it does not support simulating a step and instead will raise an Exception
    if simulating a step is attempted. To be used by tests to verify that replay from cache does not simply simulate
    again, but instead actually reads from the cache."""
    def step(self):
        raise Exception("This function is not supposed to be called during replay.")


class ModelCachableCustomFileHandling(ModelCachable):
    """ModelCachable with custom write and read implementation for the cache file. Uses a different compression
    algorithm than the default ModelCachable, which should perform slower but result in stronger compression.
    Used in a test to demonstrate the possibility of writing custom file handling."""

    def _write_cache_file(self) -> None:
        # overwrite to use different compression algorithm
        with lzma.open(self.cache_file_path, 'wb') as file:
            dill.dump(self.cache, file)

    def _read_cache_file(self) -> None:
        # overwrite to use different compression algorithm
        with lzma.open(self.cache_file_path, 'rb') as file:
            self.cache = dill.load(file)


class ModelCachableCustomSerialization(ModelCachable):
    """ModelCachable with custom state serialization and deserialization implementation for the cache.
    Instead of storing the complete model instance state, it stores just the values necessary for replay.
    In case of ModelFibonacci, storing the 'current' value is enough for replay."""

    def _serialize_state(self) -> int:
        # Store just the current value, because this is sufficient for replay. Note that for other models,
        # one could also store a list of selected attributes or anything else that is sufficient for replay.
        return self.model.current

    def _deserialize_state(self, state: int) -> None:
        # As the serialization (see function above) stores just the current value, the state that the deserialization
        # function receives is exactly this one value. So, to deserialize, it suffices to transfer that state (current
        # model value) to the model instance that is used for replay.
        self.model.current = state


class TestModelCachable(unittest.TestCase):

    def test_model_attribute_access_over_wrapper(self):
        model = ModelFibonacci()
        model = ModelCachable(model, "irrelevant_cache_file_path", CacheState.WRITE)
        assert model.running is True
        assert model.previous == 0
        assert model.custom_model_function() == 1

    def test_cache_read_fail_when_non_existing_file(self):
        model = ModelFibonacci()

        # No exception when constructing ModelCachable with CacheState.WRITE because does not try to read cache
        ModelCachable(model, "non_existing_file", CacheState.WRITE)

        # Exception when trying to construct ModelCachable with CacheState.READ and non-existing cache file
        self.assertRaises(Exception, ModelCachable, model, "non_existing_file", CacheState.READ)

    def test_cache_file_creation(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            assert not cache_file_path.is_file() and not cache_file_path.exists()
            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            for i in range(10):
                model_simulate.step()
            model_simulate.finish_run()

            assert cache_file_path.is_file()

            # assert that file created by default ModelCachable can be opened using gzip and then dill
            with gzip.open(cache_file_path, 'rb') as file:
                dill.load(file)

    def test_compare_replay_with_simulation(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
            step_count = 20
            values_simulate = []
            values_replay = []

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            for i in range(step_count):
                model_simulate.step()
                values_simulate.append(model_simulate.current)
            model_simulate.finish_run()

            # Replay
            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            for i in range(step_count):
                model_replay.step()
                values_replay.append(model_replay.current)

            # Assert that values are identical
            assert values_replay == values_simulate

    def test_cache_size(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
            step_count = 20

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            for i in range(step_count):
                model_simulate.step()
            model_simulate.finish_run()

            # Load from cache and check cache size
            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            assert len(model_replay.cache) == step_count

    def test_automatic_save_after_run_finished(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            mock_function = MagicMock(name='finish_run')
            model_simulate.finish_run = mock_function

            assert mock_function.call_count == 0

            model_simulate.run_model()

            assert mock_function.call_count == 1

    def test_replay_finish_identical_to_simulation_finish(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            model_simulate.run_model()
            final_value_simulation = model_simulate.current
            final_step_simulation = model_simulate.step_count

            # Replay
            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            model_replay.run_model()
            final_value_replay = model_simulate.current
            final_step_replay = model_replay.step_count

            assert final_step_replay == final_step_simulation
            assert final_value_replay == final_value_simulation

    def test_custom_cache_file_handling(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path_1 = Path(tmp_dir_path).joinpath("cache_file_1")
            cache_file_path_2 = Path(tmp_dir_path).joinpath("cache_file_2")

            # Simulate with regular ModelCachable
            model_1 = ModelFibonacci()
            model_1 = ModelCachable(model_1, cache_file_path_1, CacheState.WRITE)
            model_1.run_model()
            final_value_1 = model_1.current

            # Simulate with custom ModelCachable that uses stronger compression
            model_2 = ModelFibonacci()
            model_2 = ModelCachableCustomFileHandling(model_2, cache_file_path_2, CacheState.WRITE)
            model_2.run_model()
            final_value_2 = model_2.current

            # Make sure both models behaved the same way
            assert final_value_1 == final_value_2

            # Cache file 2 should be smaller than cache file 1 due to stronger compression
            assert cache_file_path_2.stat().st_size * 1.1 < cache_file_path_1.stat().st_size

    def test_custom_serialization(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path_1 = Path(tmp_dir_path).joinpath("cache_file_1")
            cache_file_path_2 = Path(tmp_dir_path).joinpath("cache_file_2")

            # Simulate with regular ModelCachable
            model_1 = ModelFibonacci()
            model_1 = ModelCachable(model_1, cache_file_path_1, CacheState.WRITE)
            model_1.run_model()
            final_value_1 = model_1.current

            # Simulate with custom ModelCachable that caches only parts of the model state that are required for replay
            model_2 = ModelFibonacci()
            model_2 = ModelCachableCustomSerialization(model_2, cache_file_path_2, CacheState.WRITE)
            model_2.run_model()
            final_value_2 = model_2.current

            # Make sure both models behaved the same way
            assert final_value_1 == final_value_2

            # Cache file 2 should be a lot smaller than cache file 1 due to storing fewer data
            assert cache_file_path_2.stat().st_size * 35 < cache_file_path_1.stat().st_size

    def test_model_cachable_optimized_precision(self):
        for precision in (1, 2, 3, 8):
            with TemporaryDirectory() as tmp_dir_path:
                cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
                step_count = 20

                # Simulate
                model_simulate = ModelFibonacci()
                model_simulate = ModelCachableOptimized(model_simulate, cache_file_path, CacheState.WRITE,
                                                        precision=precision, compress_each_step=False)
                for i in range(step_count):
                    model_simulate.step()
                model_simulate.finish_run()

                # Replay
                model_replay = ModelFibonacciForReplay()
                model_replay = ModelCachableOptimized(model_replay, cache_file_path, CacheState.READ,
                                                      compress_each_step=False)

                # The replay cache has only every precision-th step. E.g. precision is 2: only every second step.
                # 100 steps, precision 1 -> 100 cache size
                # 100 steps, precision 2 -> 50 cache size
                # 100 steps, precision 3 -> 33 cache size
                # 100 steps, precision 8 -> 12 cache size
                expected_replay_steps = step_count // precision

                assert len(model_replay.cache) == expected_replay_steps

                model_replay.run_model()
                assert model_replay.step_count == expected_replay_steps

    def test_model_cachable_optimized_compress_steps(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            # Simulate without compression
            model_no_compression = ModelFibonacci()
            model_no_compression = ModelCachableOptimized(model_no_compression, cache_file_path, CacheState.WRITE,
                                                          compress_each_step=False)
            model_no_compression.step()
            state_no_compression = model_no_compression._serialize_state()

            # Simulate with compression
            model_compression = ModelFibonacci()
            model_compression = ModelCachableOptimized(model_compression, cache_file_path, CacheState.WRITE,
                                                       compress_each_step=True)
            model_compression.step()
            state_compression = model_compression._serialize_state()

            # note that the compressed state is only slightly smaller for the example model used here, because it
            # is very small and contains almost no redundancy
            assert len(state_compression) * 1.1 < len(state_no_compression)

    def test_model_cachable_streaming_chunk_handling(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachableStreaming(model_simulate, cache_file_path, CacheState.WRITE)
            model_simulate.step()
            model_simulate.finish_run()
            value_simulate = model_simulate.current

            # Replay
            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachableStreaming(model_replay, cache_file_path, CacheState.READ)

            # manually read from stream: 1. retrieve chunk length
            chunk_length = _stream_read_next_chunk_size(model_replay.cache_file_stream)
            assert chunk_length > 0

            # 2. read the actual state data by using the given chunk length
            serialized_state = model_replay.cache_file_stream.read(chunk_length)

            # 3. set model state to deserialized chunk state
            model_replay._deserialize_state(serialized_state)

            # expect that the simulation of 1 step has the same value as replay of 1 chunk
            value_replay = model_replay.current
            assert value_replay == value_simulate

    def test_model_cachable_streaming_results(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")

            # Simulate
            model_simulate = ModelFibonacci()
            model_simulate = ModelCachableStreaming(model_simulate, cache_file_path, CacheState.WRITE)
            model_simulate.run_model()
            value_simulate = model_simulate.current

            # Replay
            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachableStreaming(model_replay, cache_file_path, CacheState.READ)
            model_replay.run_model()
            value_replay = model_replay.current

            assert value_replay == value_simulate
