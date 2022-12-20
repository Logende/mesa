import gzip
import pickle

from mesa.modelcachable import Model, ModelCachable, CacheState, ModelCachableOptimized, ModelCachableStreaming, \
    _stream_read_next_chunk_size

from unittest.mock import MagicMock
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path


class ModelFibonacci(Model):
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
    def step(self):
        raise Exception("This function is not supposed to be called during replay.")


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

            # assert that file created by default ModelCachable can be opened using gzip and then pickle
            with gzip.open(cache_file_path, 'rb') as file:
                pickle.load(file)

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

            model_simulate = ModelFibonacci()
            model_simulate = ModelCachable(model_simulate, cache_file_path, CacheState.WRITE)
            model_simulate.run_model()
            final_value_simulation = model_simulate.current
            final_step_simulation = model_simulate.step_count

            model_replay = ModelFibonacciForReplay()
            model_replay = ModelCachable(model_replay, cache_file_path, CacheState.READ)
            model_replay.run_model()
            final_value_replay = model_simulate.current
            final_step_replay = model_replay.step_count

            assert final_step_replay == final_step_simulation
            assert final_value_replay == final_value_simulation

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
                # Note that the first step is always being cached, which results in the following cache sizes:
                # 100 steps, precision 1 -> 100 cache size
                # 100 steps, precision 2 -> 50 cache size
                # 100 steps, precision 3 -> 34 cache size
                # 100 steps, precision 8 -> 13 cache size
                expected_replay_steps = 1 + (step_count - 1) // precision

                assert len(model_replay.cache) == expected_replay_steps

                model_replay.run_model()
                assert model_replay.step_count == expected_replay_steps

    def test_model_cachable_optimized_compress_steps(self):
        with TemporaryDirectory() as tmp_dir_path:
            cache_file_path = Path(tmp_dir_path).joinpath("cache_file")
            step_count = 100

            # Simulate without compression
            model_no_compression = ModelFibonacci()
            model_no_compression = ModelCachableOptimized(model_no_compression, cache_file_path, CacheState.WRITE,
                                                          compress_each_step=False)
            model_no_compression.step()
            state_no_compression = model_no_compression.serialize_state()

            # Simulate with compression
            model_compression = ModelFibonacci()
            model_compression = ModelCachableOptimized(model_compression, cache_file_path, CacheState.WRITE,
                                                       compress_each_step=True)
            model_compression.step()
            state_compression = model_compression.serialize_state()

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
            model_replay.deserialize_state(serialized_state)

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
