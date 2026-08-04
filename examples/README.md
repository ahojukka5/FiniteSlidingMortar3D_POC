# Examples

The examples are the primary interface of this proof of concept. Each one uses
the production contact solver directly and writes a small, inspectable result
set under its own `results/` directory.

## 1. Nonmatching contact patch

```bash
uv run python -m examples.contact_patch
```

This is the first completed v0.1 example. It solves two small finite-strain
TET4 bodies with one warped `QUAD4` mortar surface against two nonmatching
`TRI3` facets. See [contact_patch](contact_patch/) for the model and outputs.

## Planned v0.1 examples

- [sandwiched beam](https://github.com/ahojukka5/FiniteSlidingMortar3D_POC/issues/106)
- [rotating blocks](https://github.com/ahojukka5/FiniteSlidingMortar3D_POC/issues/24)

The large benchmark campaigns remain available for internal verification, but
they are not the user-facing workflow and are not prerequisites for running an
example.
