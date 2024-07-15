import numpy as np
import zarr
import zarr.storage


def _parse_memspec(s: str) -> int:
    """Parse a memory specification string into bytes."""
    s = s.upper()
    if s[-2:] == "KB":
        return int(s[:-2]) * 1024
    if s[-2:] == "MB":
        return int(s[:-2]) * 1024**2
    if s[-2:] == "GB":
        return int(s[:-2]) * 1024**3
    if s[-2:] == "TB":
        return int(s[:-2]) * 1024**4
    if s[-1] == "B":
        return int(s[:-1])
    return int(s)


class IdCollector:
    def __init__(self, storage: zarr.storage.Store, max_ram="1GB", **dset_kwargs):
        self._store = storage
        self._root = zarr.group(store=self._store, overwrite=True)
        self._max_ram = _parse_memspec(max_ram)
        self._dset_kwargs = dset_kwargs

        # variable state
        self._ram = 0
        self._cache = []
        self._attr_cache = []

    def add(self, key: str, id: np.ndarray, data: np.ndarray):
        self._ram += id.nbytes + data.nbytes
        self._cache.append((key, id, data))

        if self._ram > self._max_ram // 2:
            self.flush()

    def add_attr(self, full_key: str, key: str, value):
        self._attr_cache.append((full_key, key, value))

    def flush(self):
        self._ram = 0

        x: dict[int, dict[str, list[np.ndarray]]] = {}
        for key, id, data in self._cache:
            for i, d in zip(id, data):
                x.setdefault(i, {}).setdefault(key, []).append(d)
        del self._cache
        self._cache = []

        # write data
        for i, dset in x.items():
            for key, data in dset.items():
                data = np.stack(data)
                key = str(i) + "/" + key
                if key in self._root:
                    self._root[key].append(data)
                else:
                    self._root.create_dataset(key, data=data, **self._dset_kwargs)
        del x

        # write attributes
        for full_key, key, value in self._attr_cache:
            self._root[full_key].attrs[key] = value
        del self._attr_cache
        self._attr_cache = []

    def close(self):
        self.flush()

    def __del__(self):
        self.close()


class EpCollector:
    def __init__(self, *args, **kwargs):
        self._collector = IdCollector(*args, **kwargs)

        self._cur_ep = None

    def reset(self):
        self._cur_ep = self._cur_ep + 1 if self._cur_ep is not None else 0

    def add(self, key: str, data: np.ndarray):
        if self._cur_ep is None:
            self._cur_ep = 0
        self._collector.add(key, np.array([self._cur_ep]), data[np.newaxis, ...])

    def add_attr(self, full_key: str, key: str, value):
        if self._cur_ep is None:
            self._cur_ep = 0
        full_key = str(self._cur_ep) + "/" + full_key
        self._collector.add_attr(full_key, key, value)

    def flush(self):
        self._collector.flush()

    def close(self):
        self._collector.close()

    def __del__(self):
        self.close()
