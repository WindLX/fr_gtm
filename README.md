# fr_gtm

Generic Transport Model for FlyRuler. The equations, lookup tables, actuator,
wind, uncertainty, and sensor behavior are numerically aligned with
`gtm_ad`, while the FlyRuler topology is intentionally small.

## Model interface

- Physical state: 25 values
  - rigid body: position(3), body velocity(3), SO3 quaternion(4), rates(3)
  - actuator position(4)
  - Dryden filter state(8)
- Root inputs: `u(4)`, `d(4)`, `n(21)`
- Main outputs: `y(21)`, `clean_y(21)`
- Tunable parameters: wind(5) and uncertainty(28)

The system is assembled from explicit physical blocks:

```text
u ──> actuator ─┬─> aero ─┬─> eom ─┬─> sensor ──> y
                │         │        │       ▲
                └─> engine┘        │       │
d ──> wind subsystem ──────────────┘       n
mass ──────────────────────────────> eom
```

The `wind` subsystem reuses standard wind blocks for shear, steady wind, and
Dryden turbulence. `eom` reuses the standard 6-DOF rigid-body block.

## Quick start

```python
import fly_ruler_core as fr
from fr_gtm import GTMConfig, GTMSystem, build_initial_state

builder = GTMSystem(GTMConfig.default(), name="Root")
assert isinstance(builder, fr.PyIrBuilder)
model = fr.Compiler(
    fr.CompilerSpec(backend=fr.backends.numpy, integrator="rk4", dt=0.01)
).compile(builder)

x0 = build_initial_state(model)
u0 = model.u.build(
    {
        "Root::u": [0.0, 0.0, 0.0, 30.0],
        "Root::d": [0.0] * 4,
        "Root::n": [0.0] * 21,
    }
)
xdot, _, y, _ = model.continuous_fn(
    0.0, x0, model.xd.build(), u0, model.p.build()
)
print(model.y.read(y)["clean_y"])
```

All state, input, output, and parameter overrides use complete paths such as
`Root.eom::vel_body` and `Root.wind.params::wind_speed`.

## Development

```bash
just setup
just check
just test
just build
just pre-commit
```

Examples:

```bash
uv run python examples/inspect_model.py
uv run python examples/simulate.py
uv run python examples/disturbances.py
uv run python examples/trim_linearize.py
uv run python examples/backend_compare.py
```

## Reference verification

`tests/data/gtm_reference.npz` contains frozen results from the external
`gtm_ad` implementation. It covers both scales, all four flap/gear modes,
low/high Dryden turbulence, uncertainty, and sensor noise.

Regenerate it explicitly with:

```bash
uv run --with /path/to/gtm_ad \
  python scripts/generate_reference.py --gtm-ad /path/to/gtm_ad
```

Normal tests never depend on that external repository.
