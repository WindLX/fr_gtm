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
- Main outputs: `y(21)`, `clean_y(21)`, causal `feedback_y(17)`, and `clean_feedback_y(17)`
- Tunable parameters: wind(5) and uncertainty(28); `Root.wind.params::wind_direction` uses radians

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

`feedback_y` is exactly `y[:17]`: it ends at actual actuator position and excludes the four actuator-command telemetry channels. Internally connected controllers should use this causal output. The historical 21-channel vectors retain their original ordering and values for reference compatibility and logging.

All state, input, output, and parameter overrides use complete paths such as
`Root.eom::vel_body` and `Root.wind.params::wind_speed`.

Wind direction is radians clockwise from NED North for both steady wind and low-altitude Dryden rotation. Older fr_gtm revisions passed a degree-valued parameter into the radian Dryden port; the corrected `wind_low` frozen case intentionally differs from that buggy behavior.

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

The five offline examples keep their single-rate simulation flow explicit and use direct `for` loops where stepping is required. The T16000 and ActiveSidestick examples are multi-rate realtime systems, so they use std `run_async_realtime` tasks for input, dynamics, output/feedback, telemetry and recording.

`just benchmark-jax-compile` is a non-default developer benchmark for the complete RK4 GTM. It records generated source size, maximum line width, JAXPR equation counts, StableHLO sizes, and separate trace/lower/compile times for `continuous_fn` and `discrete_fn`; machine-specific JSON is written to ignored `build/jax_compile_benchmark.json` and is not frozen reference data.

Interactive stick control:

```bash
just setup-stick
just -- example-fly-stick --profile examples/fly_stick_profiles/t16000m.toml
```

ActiveSidestick controls elevator/aileron while the T16000 continues to control rudder/throttle: start `fly-ruler-server`, then run `just -- example-active-sidestick --profile examples/fly_stick_profiles/t16000m.toml --proto-address 127.0.0.1:18002`. This standalone demo uses std `run_async_realtime`, `make_discrete_model_task`, and `Recorder`; schema-first aircraft/controller/sensor/runtime telemetry is published through `fly_ruler_proto_python` and can be configured in the Proto Web before the loop begins. It does not import `fly_stick_gtm.py`. Use `--active-stick-index 1|2`, `--active-roll-limit-deg`, and `--active-pitch-limit-deg` to select and calibrate the active stick; use `--invert-roll` or `--invert-pitch` if an axis direction is reversed. Recorded data is saved under `results_active_sidestick/` unless `--no-save` is passed.

The GTM model is heavy, so the stick example defaults to the native runtime:
CasADi is used to generate a cached C++/pybind11 module, then the loop calls
that compiled module at runtime. The first run may spend extra time compiling;
later runs should reuse the cache. If the console still reports `rt < 1.0x` or
lag warnings, lower `--hz` before trying higher-fidelity settings.

The T16000 realtime path also uses std `run_async_realtime` and `Recorder`; use `--record-hz`, `--output-dir`, `--record-name`, or `--no-save` to configure recording.

For backend debugging, use `--backend numpy` or `--backend casadi`. For real-time
MSFS streaming, keep the default `--backend native`.

Dependency-free smoke test:

```bash
uv run python examples/fly_stick_gtm.py --dry-run --no-stick --no-proto --duration 0.05
```

The ActiveSidestick path has an equivalent hardware-free smoke mode after installing the `fly-stick` group:

```bash
uv run --group fly-stick python -m examples.fly_active_sidestick_gtm --dry-run --duration 0.05
```

Manual fixed-command test without a joystick:

```bash
just -- example-fly-stick --no-stick --no-proto --manual-command 10 0 0 40
```

## Ground operations

`GTMConfig.default(enable_ground=True)` adds the shared std `GroundContact` subsystem with NASA GTM T2 nose/left-main/right-main locations. It contributes contact force to the EOM, accelerometer specific force and `nz`, and exports per-wheel penetration, normal, longitudinal and lateral loads. The default disabled model keeps the existing 25-state reference topology and parameter layout; enabling ground adds only `Root.ground.params::ground_down` and ground diagnostic outputs. `build_ground_initial_state()` supports both statically supported ground roll and an airborne touchdown initial state; `forward_speed_mps` is runway-horizontal NED North speed, while `sink_rate_mps` is positive NED Down speed.

```bash
just -- example-ground --scenario takeoff --backend numpy
just -- example-ground --scenario touchdown --backend native
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

### Backend trajectory regression

`tests/data/gtm_backend_trajectory.npz` is a smaller frozen `gtm_ad` reference
used specifically to guard backend regressions. The case is the manual
`+elevator` scenario that previously exposed the CasADi/native matrix-layout
bug:

- initial altitude: `1000 m`
- initial body speed: `100 m/s`
- command: `[elevator, aileron, rudder, throttle] = [10, 0, 0, 40]`
- integrator: `rk4`
- step size: `dt = 1 / 120 s`
- samples: `t = 0, 1, 2, 3 s`

The expected physical trend is nose-down, descending, and accelerating. The
test stores `gtm_ad` samples and one-step `x_next` values in the package, so
regular test runs do not import the external `gtm_ad` checkout.

Run the normal backend test set:

```bash
uv run pytest tests/test_backends.py -q
```

This checks NumPy, CasADi, and native. JAX is optional; if it is not installed,
the JAX cases are skipped. To run the same test with a temporary JAX install:

```bash
JAX_ENABLE_X64=1 uv run --with jax --with jaxlib \
  pytest tests/test_backends.py -q -k jax
```

`JAX_ENABLE_X64=1` is intentional. Without it, JAX defaults to float32 and the
comparison becomes a precision test rather than a dynamics/backend parity test.

The test uses two layers of tolerances:

- backend-to-NumPy: strict (`1e-9` for CasADi/native, slightly looser for JAX);
- FlyRuler-to-frozen-`gtm_ad`: engineering tolerance, because FlyRuler uses its
  own SO3/RKMK integration and current GTM block decomposition while preserving
  the same model trend and numerical behavior.

Regenerate the backend trajectory reference from a local `gtm_ad` checkout with:

```bash
GTM_AD_PATH=/path/to/gtm_ad \
  uv run python tests/generate_gtm_backend_reference.py
```

If `GTM_AD_PATH` is omitted, the script looks for the sibling project path used
in this workspace: `/home/windlx/Work/Project/AircraftModels/GTM/gtm_ad`.
