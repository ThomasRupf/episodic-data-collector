# Episodic Data-Collector for (Batched) Data Collection
## TODOs
- [x] non-strict version that allows storing items that change shape over time
  - [ ] test
- [x] provide `register_key` functionality that uses certain dset kwargs for all dsets that end in the given key
  - [ ] test
- [ ] provide `zarr` and `hdf5` versions that optimize
  1. good write-speed/compression ratio
  2. good write and random-read speeds
- [ ] provide simple stat-util that allows aggregation of values over time and episodes
