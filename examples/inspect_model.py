"""Compile GTM and inspect its compact topology and path-keyed layout."""

from typing import Any

import fly_ruler_core as fr

from fr_gtm import GTMConfig, GTMSystem


def _semantic_blocks(schema: dict[str, Any]) -> list[str]:
    raw_blocks = schema.get("blocks", {})
    blocks: list[str] = []
    if isinstance(raw_blocks, dict):
        blocks.extend(
            path
            for path in raw_blocks
            if isinstance(path, str) and "._expr_" not in path and "._expr" not in path
        )
    raw_subsystems = schema.get("subsystems", [])
    if not isinstance(raw_subsystems, list):
        return sorted(blocks)
    for subsystem in raw_subsystems:
        if isinstance(subsystem, dict):
            blocks.extend(_semantic_blocks(subsystem))
    return sorted(blocks)


def main() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("GTMSystem must be instantiated as a top-level system")

    fr.dump_system_to_file(builder, "gtm_system.json")

    schema = fr.dump_system(builder)
    model = fr.Compiler(
        fr.CompilerSpec(
            backend=fr.backends.numpy, integrator="rk4", dt=0.01, output_dir="./"
        )
    ).compile(builder)

    print("Blocks:")
    for path in _semantic_blocks(schema):
        print(f"  {path}")
    print(f"\nphysical state size: {model.module.n_x}")
    print(f"SO3 tangent size:    {model.module.n_xdot}")
    print(f"parameter size:      {model.module.n_p}")
    print("\nState paths:")
    for path in model.x.keys:
        print(f"  {path}")


if __name__ == "__main__":
    main()
