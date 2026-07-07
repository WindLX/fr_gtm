# fr_gtm development notes

`fr_gtm` is the FlyRuler implementation of the Generic Transport Model.
`/home/windlx/Work/Project/AircraftModels/GTM/gtm_ad` is the numerical
reference; preserve its equations and table orientation.

The runtime topology must remain four semantic blocks:

- `Actuator`: four first-order actuator states.
- `Wind`: steady wind and eight Dryden states.
- `Airframe`: rigid body, uncertainty, atmosphere, propulsion and aerodynamics.
- `Sensor`: clean output assembly and sensor errors.

Keep intermediate formulas as pure Signal helpers. Do not reintroduce fanout,
sum, extractor, proxy-port, or one-formula subsystems.

Aerodynamic data is packaged in `src/fr_gtm/data/aero_tables.npz`; runtime code
must not depend on SciPy or MATLAB files. Use complete FlyRuler paths in
examples, trim, linearization and tests.

Validation:

```bash
just pre-commit
just build
uv run python examples/backend_compare.py
```

The frozen `gtm_ad` cases are generated only through
`scripts/generate_reference.py`; normal tests must remain self-contained.
