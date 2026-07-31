# Contributing

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) with a small
set of additional structure rules. Commits should be **small and targeted**:
change a small number of files that form one logical group describable by a single
coherent message. Prefer several focused commits over one large mixed commit.

### Message structure

```
<type>: <topic>

<1-3 sentence summary of what the change does and why>

- <optional detail>
- <optional detail>
```

1. **Topic line** — a conventional-commit type (`feat`, `fix`, `docs`, `chore`,
   `refactor`, `test`, `build`, `ci`, `perf`, `style`) followed by an imperative,
   lower-case summary. Keep the whole topic line **under 72 characters**. No
   trailing period.
2. **Blank line**, then a **brief description**: 1-3 sentences summarising the
   change. Say what changed and, when useful, why.
3. If further detail is warranted, a **blank line** then a **bulleted list** of the
   specifics.
4. **Wrap the body at under 80 characters** per line.
5. **No trailers** (no `Co-Authored-By`, no `Signed-off-by`, etc.).

### Example

```
feat: add sparse LU backend for reduced free-free systems

Solve the strong-Dirichlet reduced CSR block with SciPy sparse LU instead of a
dense conversion, and record setup/solve timings alongside the existing
residual diagnostics.

- Fall back to the dense backend when SciPy is unavailable.
- Add focused backend tests comparing sparse and dense solutions.
```

### Scope of a commit

A commit's files should relate to each other. Good groupings, for example:

- a module and the config keys or options it introduces;
- a formulation note and the code it documents;
- a feature and its tests.

Avoid mixing unrelated changes (e.g. a new feature and an unrelated docs fix) in
one commit.
