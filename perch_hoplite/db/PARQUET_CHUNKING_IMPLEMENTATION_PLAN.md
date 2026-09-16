# Implementation plan: optional chunked Parquet files per deployment

Companion to `PARQUET_DUCKDB_PLAN.md` Q9 ("Chunked score files and
`recordings.score_chunk_path`"), which was designed but **not built** —
the shipped `ParquetScoresDB` (`perch_hoplite/db/parquet_scores_impl.py`)
writes exactly one Parquet file per deployment, as a whole-file replacement.
This doc is the concrete, code-level plan for building Q9's design as an
**opt-in extension** of what already exists, not a rewrite: a caller who
keeps calling `insert_deployment_scores()` once per deployment must see
zero behavior change.

Read `PARQUET_DUCKDB_PLAN.md`'s "Implementation status" section and Q9
first — this doc assumes that context and doesn't repeat the reasoning,
only the mechanics of building it against the real code.

## Ground truth (current shipped state)

- One Parquet file per deployment: `deployment_parquet_path(deployment_id)`
  → `scores_dir / f'deployment_{deployment_id}.parquet'`
  ([parquet_scores_impl.py:464-466](parquet_scores_impl.py#L464-L466)).
- Exactly **6 call sites** reference `deployment_parquet_path`, all in
  `parquet_scores_impl.py`:
  - [L1011](parquet_scores_impl.py#L1011) — `remove_deployment()`: deletes
    the file outright.
  - [L1731](parquet_scores_impl.py#L1731) — `insert_deployment_scores()`:
    checks existence for `handle_existing='error'`, then writes.
  - [L1801](parquet_scores_impl.py#L1801) — `_scores_union_query()`: one
    `read_parquet(...)` branch per deployment, `UNION ALL`'d.
  - [L1844](parquet_scores_impl.py#L1844) —
    `_scores_union_query_all_classes()`: same shape, plus `UNPIVOT`.
  - [L2560](parquet_scores_impl.py#L2560) — `get_scores_batch()`: single-file
    `read_parquet(?)` joined against a keys table.
- `recordings` table DDL is in `_setup_tables()`
  ([parquet_scores_impl.py:505-514](parquet_scores_impl.py#L505-L514)) — no
  `score_chunk_path` column yet.
- `insert_deployment_scores()` signature: `(deployment_id, recording_ids,
  start_times, end_times, scores, handle_existing: Literal['error',
  'overwrite'] = 'error')`
  ([parquet_scores_impl.py:1664-1672](parquet_scores_impl.py#L1664-L1672)).
  Writes via temp-file + atomic rename
  ([L1749-1752](parquet_scores_impl.py#L1749-L1752)) — this crash-safety
  property must be preserved per-chunk.
- Tests: `perch_hoplite/db/tests/parquet_scores_impl_test.py` (~1,500 lines).
  Existing `handle_existing` coverage: L970-1090 (shape/length validation,
  error-on-existing, overwrite) and L1348-1461 (scores + detections
  interplay). No `parquet_scores` entry in `db/tests/test_utils.py`'s
  `DB_TYPES`/`make_db()` — this backend's tests are standalone, not run
  through the cross-backend conformance suite.

## Design recap (what we're building)
- there is still only one deployment in each parquet, but there may be multiple parquet "chunk files" per deployment. The user can decide to chunk inference and write parquets with subsets of a deployment
- `recording_ids`/`start_times`/`end_times`/`scores` passed to
  `insert_deployment_scores()` in one call always land in **one new chunk
  file** — never merged into an existing chunk, never split across chunks.
  Chunk boundaries are entirely a function of how the caller batches their
  calls; the method itself does no internal splitting.
- Chunk files are auto-named/numbered internally (e.g.
  `deployment_{id}_{n}.parquet`), not caller-chosen.
- `recordings.score_chunk_path` (new nullable column in `recordings` table) is the source of
  truth for "does this recording have scores yet, and if so where."
- New `handle_existing_recordings: Literal['skip', 'error', 'overwrite'] =
  'skip'` parameter. The name is deliberate: the comparison/dedup logic is
  keyed **per recording** (via `score_chunk_path`, a `recordings`-table
  column), never per score row. A recording either already has scores
  written somewhere or it doesn't — there is no partial state where some of
  a recording's rows exist and others don't, since a recording's scores are
  always written together in whichever call first wrote them. Concretely:
  if `handle_existing_recordings='overwrite'` and the incoming batch
  contains any rows for recording 1, **every** existing row for recording 1
  (all of its start/end-time windows, wherever they live) is replaced by
  whatever recording 1's rows are in the incoming batch — not a row-by-row
  diff/merge. (Distinct from the existing whole-file `handle_existing` —
  see "Naming/API shape" below for how these coexist.)
- Query paths (`_scores_union_query`, `_scores_union_query_all_classes`,
  `get_scores_batch`) glob/list all of a deployment's chunk files instead of
  assuming exactly one.
- `remove_deployment()` deletes all chunk files for the deployment, not one.

## Naming/API shape decision (resolve before coding)

The existing `handle_existing='error'|'overwrite'` is a **whole-file**
concept (does *the* Parquet file already exist). Q9's `on_existing`/`skip`
idea is a **per-recording** concept once there can be multiple chunk files.
These are not the same question once chunking exists, and need one coherent
parameter surface rather than two overlapping ones. Recommended resolution:

- Keep a single parameter, rename to `handle_existing_recordings:
  Literal['error', 'skip', 'overwrite'] = 'error'` (keep `'error'` as
  default to preserve today's behavior exactly — do not silently change the
  default to `'skip'`, since that would change behavior for existing
  callers who rely on `'error'` catching accidental double-writes). The
  name is chosen over the shorter `handle_existing` to make clear, at the
  call site and in the signature itself, that this checks whether each
  **recording** already has scores — not whether individual score rows
  match — since that distinction is easy to assume wrong once multiple
  chunk files exist.
- Semantics, defined per recording in the incoming batch against its
  current `score_chunk_path` (NULL = never scored):
  - `'error'`: if **any** recording in the batch already has a non-NULL
    `score_chunk_path`, raise before writing anything (matches today's
    whole-file check, generalized to per-recording — for a
    single-call-per-deployment caller with no prior chunks, this is
    identical to today's behavior since every recording's
    `score_chunk_path` is NULL).
  - `'skip'`: drop recordings with a non-NULL `score_chunk_path` from the
    batch before writing; write a chunk containing only the
    never-before-scored recordings. If the filtered batch is empty, write
    nothing (no zero-row chunk file) — decide and document this edge case
    explicitly.
  - `'overwrite'`: for recordings already present, update their rows
    **in place** within their existing chunk file(s) (see "Overwrite
    implementation" below) — `score_chunk_path` for those recordings does
    not change. Only the genuinely-new (never-before-scored) recordings in
    the batch go into a newly-written chunk.
- Confirm this naming/default choice with the user before implementing —
  it's a small API break (`handle_existing` → `handle_existing_recordings`, default
  `'error'`→`'error'` unchanged but meaning generalized) worth a deliberate
  sign-off rather than assuming.

## Work items

### 1. Schema migration: `recordings.score_chunk_path`
- Add `score_chunk_path VARCHAR` (nullable) to the `CREATE TABLE recordings`
  DDL in `_setup_tables()`
  ([parquet_scores_impl.py:505-514](parquet_scores_impl.py#L505-L514)).
- `_setup_tables()` is guarded by `if 'recordings' in existing: return`
  ([L483-484](parquet_scores_impl.py#L483-L484)) — i.e. it only runs DDL on
  a brand-new `.duckdb` file. Existing on-disk databases created before this
  change will **not** get the new column automatically. Decide: is an
  `ALTER TABLE recordings ADD COLUMN IF NOT EXISTS score_chunk_path VARCHAR`
  migration step needed for pre-existing test/dev databases, or is
  "recreate the DB" acceptable at this stage (check with user — likely fine
  pre-1.0, but confirm no real user data exists in this format yet)?
- Populate `score_chunk_path` retroactively for any deployment that already
  has a single `deployment_{id}.parquet` file under the current scheme —
  needed so existing single-file deployments interoperate with the new
  per-recording lookup once chunking ships. Could be a one-time backfill
  function or handled lazily (treat "no `score_chunk_path` but a legacy
  `deployment_{id}.parquet` exists" as an implicit single chunk at query
  time) — pick one and be consistent across all four query/write call
  sites.

### 2. Chunk file naming and discovery
- Add a helper, e.g. `_deployment_chunk_paths(deployment_id) ->
  list[epath.Path]`, that lists existing chunk files for a deployment —
  either by globbing `scores_dir` for `deployment_{id}_*.parquet` (plus the
  legacy `deployment_{id}.parquet` if not yet migrated) or, more robustly,
  by `SELECT DISTINCT score_chunk_path FROM recordings WHERE deployment_id
  = ?` (preferred — avoids relying on the filesystem being consistent with
  the DB, matches Q9's design intent that `recordings` is the source of
  truth, not directory listing).
- Add `_next_chunk_path(deployment_id) -> epath.Path` that picks the next
  unused suffix. If chunk discovery goes through `recordings` (previous
  bullet) rather than a directory glob, this needs a tie-breaker for the
  rare case a chunk file exists on disk but isn't yet referenced by any
  `recordings` row (e.g. a previous write crashed after the atomic rename
  but before the `recordings` update — see work item 4's crash-ordering
  note). Simplest safe approach: glob the directory for the max existing
  numeric suffix and add 1, regardless of what `recordings` currently
  references — cheap (one directory listing per write) and never collides.

### 3. Rewrite `insert_deployment_scores()`
([parquet_scores_impl.py:1664-1752](parquet_scores_impl.py#L1664-L1752))
- Replace the `parquet_path.exists()` whole-file check
  ([L1731-1736](parquet_scores_impl.py#L1731-L1736)) with a per-recording
  lookup: `SELECT id, score_chunk_path FROM recordings WHERE id IN (...)`
  for the incoming `recording_ids`.
- Branch on `handle_existing_recordings` per the semantics in "Naming/API
  shape" above. For `'skip'`, filter `recording_ids`/`start_times`/
  `end_times`/`scores` (the numpy array — filter by boolean mask, keyed on
  `recording_id` membership so every row belonging to a skipped recording
  is dropped together, preserving row order for the rest) down to the
  never-scored subset before proceeding.
- Write path stays the same shape as today (`_build_scores_arrow_table` →
  `pq.write_table` to a `.tmp` path → atomic `.replace()`,
  [L1738-1752](parquet_scores_impl.py#L1738-L1752)), just targeting
  `_next_chunk_path(deployment_id)` instead of the fixed
  `deployment_parquet_path(deployment_id)`.
- After the atomic rename succeeds, update
  `UPDATE recordings SET score_chunk_path = ? WHERE id IN (...)` for every
  recording actually written in this chunk — this is the "write file, then
  update `recordings`" ordering from Q9, so a crash between the two leaves
  an orphaned-but-harmless file rather than a dangling reference.
- Empty-batch edge case (all recordings skipped, or caller passes an empty
  batch): return without writing a file or touching `recordings` — avoid
  zero-row chunk files, which would need special-casing in every query path
  and in `_next_chunk_path`'s numbering.

### 4. Overwrite implementation (the expensive path)

Update-in-place, not move-to-new-chunk: an overwritten recording's
`score_chunk_path` never changes, only the row values within its existing
chunk file do. Rejected alternative (moving overwritten rows into the newly
written chunk, leaving the old chunk permanently smaller) was considered
and dropped — it costs the same (reading/rewriting every affected old chunk
is unavoidable either way, and dominates), but adds churn the in-place
version avoids for free: `score_chunk_path` updates for the overwritten
subset, and old chunks that shrink a little more each time a recording in
them gets re-scored, eventually needing empty-file cleanup. In-place update
sidesteps both.

- For `handle_existing_recordings='overwrite'`: group the batch's already-scored
  recordings by their current `score_chunk_path`. For each affected chunk
  file, read it back (`read_parquet`), replace the rows belonging to the
  recordings being overwritten with their new values (leaving every other
  row in that chunk untouched), and rewrite that chunk file in place (same
  temp+rename pattern). `score_chunk_path` for these recordings is
  unchanged — no `recordings` update needed for this subset.
- Remove the overwritten recordings from the batch, then proceed with a
  normal new-chunk write (work item 3) for whatever remains — i.e. only the
  genuinely-new (never-before-scored) recordings in the batch, if any, get
  a newly-written chunk and a `score_chunk_path` update. If every recording
  in the batch already existed, no new chunk is written at all (same
  empty-batch handling as work item 3's `'skip'` path).
- Note in the docstring, as Q9 specifies, that cost here is bounded by how
  fragmented the affected recordings' prior chunks are, not by the new
  batch's size — carry forward the existing "IMPORTANT" docstring block
  ([L1681-1696](parquet_scores_impl.py#L1681-L1696)) and adapt its wording
  from "whole deployment" to "affected chunk(s)."

### 5. Query paths: glob/list instead of one path
- `_scores_union_query()` ([L1774-1815](parquet_scores_impl.py#L1774-L1815))
  and `_scores_union_query_all_classes()`
  ([L1817-1857](parquet_scores_impl.py#L1817-L1857)): replace the single
  `parquet_path.exists()` check + one `read_parquet('{path}')` branch per
  deployment with either (a) one branch per *chunk file* (simplest,
  mechanical change — just iterate `_deployment_chunk_paths(deployment_id)`
  instead of a single path and add one `UNION ALL` branch per chunk), or
  (b) one branch per deployment using DuckDB's native multi-file support
  (`read_parquet(['path1.parquet', 'path2.parquet', ...])` or a glob
  string) — fewer `UNION ALL` branches overall, likely better for DuckDB's
  query planner when a deployment has many chunks. Prefer (b); benchmark
  against (a) using the existing
  `parquet_scores_class_scaling_test.ipynb`/`parquet_scores_stress_test.ipynb`
  notebooks if there's any doubt, since this repo already has
  measured-not-assumed performance conclusions in "Implementation status."
- `get_scores_batch()` ([L2526-2601](parquet_scores_impl.py#L2526-L2601)):
  the `LEFT JOIN read_parquet(?)` at
  [L2577](parquet_scores_impl.py#L2577) needs the same
  single-path-to-multi-path change. Since this method already looks up
  specific `(recording_id, start_time, end_time)` keys, consider resolving
  each requested recording's `score_chunk_path` first and only reading the
  chunk(s) actually needed (a real pruning win noted in Q9 — "resolves
  `score_chunk_path` directly per recording and reads only the relevant
  chunk(s)") rather than globbing every chunk in the deployment regardless
  of which recordings were requested.

### 6. `remove_deployment()`
([parquet_scores_impl.py:972-1013](parquet_scores_impl.py#L972-L1013))
- Replace the single `parquet_path.exists() / unlink()`
  ([L1011-1013](parquet_scores_impl.py#L1011-L1013)) with: collect all
  chunk paths for the deployment (via `_deployment_chunk_paths`, called
  *before* the `recordings` rows are deleted at
  [L1001](parquet_scores_impl.py#L1001), since that's the lookup source),
  then unlink each. Keep the `if exists` guard per file (a chunk file could
  already be gone from a prior partial failure).

### 7. Legacy single-file compatibility
- Decide explicitly (don't let this fall out by accident): does the new
  code path need to keep reading a bare `deployment_{id}.parquet` (no chunk
  suffix) written by the pre-chunking code, or is a migration/rename step
  required first? Simplest: treat `deployment_{id}.parquet` (no suffix) as
  chunk `0` for discovery purposes, so old and new data coexist without a
  migration. State this explicitly in `_deployment_chunk_paths` and in the
  module docstring.

### 8. Tests (`perch_hoplite/db/tests/parquet_scores_impl_test.py`)
- Update existing `handle_existing` tests (L970-1090, L1348-1461) for the
  renamed/generalized `handle_existing_recordings` parameter — confirm `'error'`'s
  single-call-per-deployment behavior is provably unchanged (this is the
  regression bar: every existing test in this file should pass with at
  most a rename, no behavior change, when a deployment is only ever written
  once).
- New tests to add:
  - Multiple `insert_deployment_scores()` calls for the same deployment
    with disjoint recordings → all chunks queryable together via
    `select_scores`/`get_scores_batch`/`add_detections_from_threshold`.
  - `handle_existing_recordings='skip'` with partial overlap (some recordings new, some
    already scored) → only new recordings' scores land in the new chunk;
    old scores for previously-scored recordings are unchanged and still
    queryable.
  - `handle_existing_recordings='error'` with partial overlap → raises, nothing written
    (verify no new chunk file appears on disk after the raise).
  - `handle_existing_recordings='overwrite'` spanning recordings from *different* prior
    chunks → confirm each old chunk is correctly rewritten in place with
    the new values (or deleted if emptied), `score_chunk_path` for those
    recordings is unchanged, and no new chunk is written when the batch
    contains no genuinely-new recordings.
  - `handle_existing_recordings='overwrite'` with a mix of already-scored and
    never-scored recordings in one batch → old chunk(s) updated in place
    for the former, a new chunk written containing only the latter.
  - `remove_deployment()` with 3+ chunk files → confirm all are deleted,
    none orphaned.
  - Legacy single-file deployment (write via a simulated old-style path, or
    a fixture) → confirm it's still readable post-migration (work item 7).
  - Empty-batch / all-skipped edge case → confirm no zero-row chunk file is
    created and no error.
- Consider whether this is also the moment to add the `parquet_scores`
  entry to `db/tests/test_utils.py`'s `DB_TYPES`/`make_db()` (existing gap,
  noted in `PARQUET_DUCKDB_PLAN.md`'s "Suggested next steps" item 4) — not
  required for chunking itself, but chunking touches `remove_deployment`
  and `recordings`, which the cross-backend conformance suite exercises for
  other backends; worth a judgment call on scope rather than silently
  bundling it in.

## Suggested split for parallel work

If two agents split this:
- **Agent A**: work items 1-4 (schema, naming/discovery, write path,
  overwrite) — the mutation side. Lands `score_chunk_path`, the rewritten
  `insert_deployment_scores`, and `remove_deployment`'s update (item 6,
  small and naturally follows from item 2's discovery helper).
- **Agent B**: work item 5 (query paths) — can be developed against a
  *fake* multi-chunk fixture (manually write 2-3 small Parquet files with
  today's schema and set `score_chunk_path` by hand in a test DB) without
  waiting on Agent A's write path to land, then re-verified once real.

Both should coordinate on work item 8 (tests) and the "Naming/API shape"
decision up front, since that decision (parameter name, default, exact
`'skip'` semantics) affects both sides' signatures.
