# Examples

The examples are the primary interface of this proof of concept. Each one uses
the production contact solver directly and writes a small, inspectable result
set under its own `results/` directory.

## 1. Nonmatching contact patch

```bash
uv run python -m examples.contact_patch
```

The smallest completed v0.1 example solves two small finite-strain TET4 bodies
with one warped `QUAD4` mortar surface against two nonmatching `TRI3` facets.
See [contact_patch](contact_patch/) for the model and outputs.

## 2. Nonmatching sandwiched beam

```bash
uv run python -m examples.sandwiched_beam
```

The second v0.1 example applies normal preload and then bends two TET4 beams
through a nonmatching `QUAD4`/`QUAD4` mortar interface. It also solves one
conforming monolithic reference response. See
[sandwiched_beam](sandwiched_beam/) for the model, checks, and limitations.

## Planned v0.1 example

- [rotating blocks](https://github.com/ahojukka5/FiniteSlidingMortar3D_POC/issues/24)

The large benchmark campaigns remain available for internal verification, but
they are not the user-facing workflow and are not prerequisites for running an
example.
