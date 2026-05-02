Phase 0 bootstraps the Dataset-A builder without pulling in heavyweight runtime dependencies yet. I will create the requested `dataset_builder/` layout, fill the four core YAML configs, and add importable Python modules for assets, simulation, rendering, token extraction, pair sampling, split generation, and single-episode orchestration.

The smoke test will verify that internal modules import cleanly, the configs exist, deterministic per-episode seeding works, and the default PartNet-Mobility path is wired to `../CAP-A2GN/data/raw_data/partnet-mobility/`. Simulator and Blender integrations will be lazy imports so a fresh environment can pass Phase 0 even before PyBullet, Taichi/Warp, Blender Python, or 3DGS tooling are installed.

If any external dependency is missing, Phase 0 will report it as optional readiness status rather than failing. Real simulator validation begins in Phase 1.
