import tempfile

import numpy as np
import zarr

from collector import BatchedEpCollector, EpCollector, IdCollector


def test_id_collector():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        collector = IdCollector(store, max_ram="1MB")

        # Add data of various shapes and dtypes
        collector.add("data1", np.array([1, 2, 3]), np.array([10, 20, 30]))
        collector.add("data2", np.array([1, 2]), np.array([[1.1, 2.2], [3.3, 4.4]]))
        collector.add_attr("1/data1", "attr1", "value1")
        collector.flush()

        # Verify the data
        root = zarr.open(store)
        assert np.array_equal(root["1/data1"][:], np.array([10]))
        assert np.array_equal(root["2/data1"][:], np.array([20]))
        assert np.array_equal(root["3/data1"][:], np.array([30]))
        assert np.array_equal(root["1/data2"][:], np.array([[1.1, 2.2]]))
        assert np.array_equal(root["2/data2"][:], np.array([[3.3, 4.4]]))
        assert root["1/data1"].attrs["attr1"] == "value1"


def test_ep_collector():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        collector = EpCollector(store, max_ram="1MB")

        # Add data of various shapes and dtypes
        collector.add({"data1": np.array([10, 20, 30])})
        collector.reset()
        collector.add({"data2": np.array([1.1, 2.2, 3.3])})
        collector.add_attr("data2", "attr1", "value1")
        collector.flush()

        # Verify the data
        root = zarr.open(store)
        assert np.array_equal(root["0/data1"][:], np.array([[10, 20, 30]]))
        assert np.array_equal(root["1/data2"][:], np.array([[1.1, 2.2, 3.3]]))
        assert root["1/data2"].attrs["attr1"] == "value1"


def test_batched_ep_collector():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        collector = BatchedEpCollector(3, store, max_ram="1MB")

        # Add data of various shapes and dtypes
        collector.add({"data1": np.array([[1, 2], [3, 4], [5, 6]])})
        collector.reset(np.array([0, 2]))
        collector.add({"data2": np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])})
        collector.add_attr("data2", "attr1", np.array(["value1", "value2", "value3"]))
        collector.flush()

        # Verify the data
        root = zarr.open(store)
        assert np.array_equal(root["0/data1"][:], np.array([[1, 2]]))
        assert np.array_equal(root["1/data1"][:], np.array([[3, 4]]))
        assert np.array_equal(root["2/data1"][:], np.array([[5, 6]]))
        assert np.array_equal(root["3/data2"][:], np.array([[1.1, 2.2]]))
        assert np.array_equal(root["1/data2"][:], np.array([[3.3, 4.4]]))
        assert np.array_equal(root["4/data2"][:], np.array([[5.5, 6.6]]))
        assert root["3/data2"].attrs["attr1"] == "value1"
        assert root["1/data2"].attrs["attr1"] == "value2"
        assert root["4/data2"].attrs["attr1"] == "value3"


if __name__ == "__main__":
    test_id_collector()
    test_ep_collector()
    test_batched_ep_collector()
    print("All tests passed!")
