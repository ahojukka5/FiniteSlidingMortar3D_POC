# Examples

The examples are the primary interface of this proof of concept. Each one uses
the production contact solver directly and writes a small, inspectable result
set under its own `results/` directory.

## 1. Nonmatching contact patch

```bash
uv run python -m examples.contact_patch
```

The smallest v0.1 example solves two small finite-strain TET4 bodies with one
warped `QUAD4` mortar surface against two nonmatching `TRI3` facets. See
[contact_patch](contact_patch/) for the model and outputs.

## 2. Nonmatching sandwiched beam

```bash
uv run python -m examples.sandwiched_beam
```

The second v0.1 example applies normal preload and then bends two TET4 beams
through a nonmatching `QUAD4`/`QUAD4` mortar interface. It also solves one
conforming monolithic reference response. See
[sandwiched_beam](sandwiched_beam/) for the model, checks, and limitations.

## 3. Rotating blocks

```bash
uv run python -m examples.rotating_blocks
```

The large-sliding example compresses, translates, and rotates a smaller TET4
block over a larger block while the nonmatching mortar overlap topology
changes. See [rotating_blocks](rotating_blocks/) for the prescribed path,
outputs, and limitations.

The large benchmark campaigns remain available for internal verification, but
they are not the user-facing workflow and are not prerequisites for running an
example.
