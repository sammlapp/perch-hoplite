# Plan: Parquet + DuckDB/SQLite backend for wide per-window class scores

## Implementation status (current)

`perch_hoplite/db/parquet_scores_impl.py` is implemented and diverges from
this plan in one deliberate simplification, plus has grown a couple of
capabilities beyond what's written below:

- **No chunking.** Q9's chunked-score-files design (`score_chunk_path`,
  multiple chunk Parquet files per deployment, `on_existing='skip'/'error'/
  'overwrite'`) was **not built**. The shipped implementation is simpler:
  exactly one Parquet file per deployment
  (`deployment_parquet_path(deployment_id)`), written by
  `insert_deployment_scores(..., handle_existing='error'|'overwrite')` as a
  **whole-file replacement** — see that method's docstring. `'overwrite'`
  replaces the entire file with exactly the rows passed in (not a merge);
  there is no partial/incremental append and no chunk numbering. This keeps
  `remove_deployment()` a plain file delete (matches Q6 as planned) and
  keeps every query a single `read_parquet()` per deployment rather than a
  glob over chunks. Revisit only if a real deployment turns out too large
  to hold in memory for one write (Q9's original motivation).
- **Query API is `select_scores`/`select_detections`, not
  `get_class_scores`.** What Q4 sketched as a single `get_class_scores(...)`
  became a considerably richer `select_scores(class_name, selection='top'|
  'random'|'binned', by_deployment=..., date_bins=..., deployments_filter=...,
  recordings_filter=..., min_score=..., max_score=..., ...)` — one query
  composing selection mode, independent stratification, and filters, plus
  a parallel `select_detections()` for the `detections` table. See their
  docstrings for the full parameter set and examples.
- **`get_scores_batch(deployment_id, recording_ids, start_times,
  end_times)`** was added beyond the original plan: fetches full
  `N_classes`-wide score vectors for specific `(recording, offsets)` keys
  (e.g. context around a detection), independent of `select_scores`'s
  single-class-column path.
- **Empirical finding: single-class lookup scales gently with class
  count, not with data volume.** `parquet_scores_class_scaling_test.ipynb`
  benchmarks `select_scores` for one class against DBs with 1/10/100/1,000
  total classes, holding row count fixed. At small Parquet files (100
  deployments × 1,000 windows), the slowdown from 1 to 1,000 classes was
  ~10x (absolute times still small, ~13ms → ~130ms) — attributed to
  per-file Parquet footer/schema metadata parsing scaling with column
  count, reparsed on every deployment file scan, even though DuckDB
  correctly projects only the target column (confirmed via
  `_scores_union_query`, which selects the single class column, not `*`).
  Re-run with more realistic sizes (100k windows/deployment), the
  slowdown at 1,000 classes dropped to only ~2x — i.e. the fixed
  per-column metadata cost is amortized against a larger per-file row
  count, so it matters least exactly when it would otherwise hurt most.
  **Net: lookups stay fast whether the DB is small (fast regardless of
  class count) or large-with-many-classes (metadata overhead is a small
  fraction of a bigger scan)** — the class-count scaling cost is real but
  not a practical concern at realistic deployment sizes.

## Next Steps
1. ~~consider making the recording rather than the deployment the atomic unit
   for addition~~ — RESOLVED. Considered a chunked-score-files design (see
   Q9 below, "Chunked score files and `recordings.score_chunk_path`"), but
   the shipped implementation is simpler: one Parquet file per deployment,
   written whole-file by `insert_deployment_scores()` (see "Implementation
   status" above). Revisit Q9's chunking design only if a real deployment
   proves too large to hold in memory for one write.
2. add documentation in db/README.md of the interface design/methods/usage examples


## Motivation

Hoplite's existing backends (`in_mem_impl`, `sqlite_usearch_impl`) are built
around one dense embedding vector per window, with USearch providing
indexed/approximate nearest-neighbor search over those vectors. Our use case
is different in two ways described in the request:

1. We store **per-window model output scores over a list of classes**
   (10–500 typically, up to ~10k), not a single embedding vector.
2. We do **not** need indexed/approximate vector similarity search. Access is
   overwhelmingly "give me all values (across windows/files) for class X,
   optionally filtered/joined with deployment or recording metadata" — a
   columnar, analytical access pattern, not a nearest-neighbor query.

The sketch in `Proposed Prediction Storage Strategy.md` and the notebook
(`try_parquet_and_db_storage.ipynb`) validate the basic mechanics: wide
Parquet per deployment (rows = clips, columns = classes), DuckDB tables for
relational metadata (`points`/`deployments`/`files`), and DuckDB's
`parquet_scan()` to join metadata with scores and to prune columns/row groups
for a single-class query.

This doc proposes how to fold that pattern into Hoplite's existing
`HopliteDBInterface` shape (deployments / recordings / windows / annotations
/ hoplite_metadata) so that existing tooling (annotation ingestion, filters,
`db_loader`, labeling workflows) keeps working, while getting Parquet's
columnar pruning for the score data itself.

## Recommendation: extend, not rewrite — new module, pure DuckDB

Follow the structure of `sqlite_usearch_impl.py` (identifier validation,
filter → WHERE-clause compilation, DDL, extra-column handling) as a reference
implementation, but re-derive it in DuckDB's SQL dialect rather than reusing
`sqlite3` — see Q1 (resolved: pure DuckDB, no SQLite at all, so that a single
writer process/connection can serve both metadata mutation and Parquet-joined
reads without juggling two engines). Write a **new module**, e.g.
`perch_hoplite/db/parquet_scores_impl.py`, that:

- Keeps `deployments`, `recordings`, `annotations`, `hoplite_metadata` as
  native DuckDB tables, structurally similar to `sqlite_usearch_impl`'s
  tables.
- Has **no `windows` table at all** (see "Rethinking `windows`" below) —
  scores live entirely in per-deployment Parquet files, addressed by
  `(recording_id, start_time, end_time)`, with no synthetic per-clip id
  minted or stored anywhere.
- Adds a new **`detections` table**: sparse, score-thresholded
  single-class hits, replacing the role `windows` plays in other backends
  as the id-bearing thing that can be referenced/annotated/reviewed.
- Drops indexed ANN search entirely (`get_embedding`/`get_embeddings_batch`/
  `search` all raise `NotImplementedError` — see Q4) in favor of new,
  explicitly-named score-query methods.

This reuses the *shape* of the existing implementation (same filter-dict
semantics, same table-extension mechanism) even though the SQL layer itself
must be rewritten for DuckDB, and lets us reuse the existing conformance test
suite (`db/tests/hoplite_test.py`, `interface_test.py`, parametrized over
`DB_TYPES` in `tests/test_utils.py`) for the parts of the interface that still
apply (deployments/recordings/annotations/metadata), while accepting that the
window/embedding-centric tests will need to be skipped or reworked for this
backend.

Implement it as its own dataclass implementing `HopliteDBInterface`
(mirroring the structure of `SQLiteUSearchDB`) rather than subclassing it,
since nearly everything window/embedding/USearch-specific either doesn't
exist in this backend or needs to be replaced by detection-based logic.

## Rethinking `windows`: no per-clip table, add `detections` instead

The original plan kept a slimmed `windows` table (one row per clip, minted
id, pointer into Parquet) purely so Hoplite's id-based conventions
(`get_window`, annotation linkage) kept working. At realistic scale (10k
classes × 1M clips), that table would have 1M rows whose *only* purpose is
bookkeeping — `(recording_id, start_time, end_time)` already uniquely
addresses a score row in Parquet, so a synthetic `window_id` is pure
indirection with no consumer. Nothing needs to reference an individual
score-matrix cell by id; the whole point of per-deployment Parquet is to
let DuckDB scan/project/prune over all of them at once, not to look up one
row at a time.

What *does* need an id-bearing table is the sparse, discrete output of
applying a score threshold: e.g. scanning a 10,000-class × 1M-clip Parquet
file for `score > threshold` might yield ~10,000 `(recording, start_time,
end_time, class, score)` "detection" tuples — small enough to be a real
table, and the kind of thing a human might later review, confirm, or turn
into an annotation, so it benefits from having a stable id the way
`annotations` rows do.

**`detections` table** (DuckDB-native, replaces `windows` as the
id-bearing/reviewable unit):
- `id` — autoincrement primary key.
- `recording_id` — FK to `recordings`.
- `start_time`, `end_time` — offsets of the detection window.
- `class_name` — the single class this detection is for.
- `score` — the score that triggered it.
- extra columns (provenance, threshold used, model run, etc.) via the same
  `add_extra_table_column()` mechanism as other tables.

Populated via a bulk operation, e.g. `add_detections_from_threshold(
class_name, threshold, deployments_filter=..., recordings_filter=...)`,
which runs a DuckDB query directly over the relevant Parquet scan(s) and
inserts qualifying rows — or via direct bulk insert if the caller already
has precomputed tuples. `annotations` stays independent (human labels, not
required to originate from a detection), though a `detection_id` could
optionally be added as a provenance/lineage link later if useful.

Interface fallout: `insert_window()`, `get_window()`, `remove_window()`,
`get_window_annotations()` all raise `NotImplementedError` (there is no
per-window row to address). New backend-specific methods
(`insert_detection`/`insert_detections_batch`, `get_detection`,
`remove_detection`, `get_detection_annotations`) fill the equivalent role for
detections, mirroring the existing annotation API shape rather than the
window API shape.

## Proposed structure

**Relational store (DuckDB-native tables):**
- `hoplite_metadata` — unchanged in spirit (key-value config, e.g. fixed
  class vocabulary — see Q5).
- `deployments` — unchanged (`id`, `name`, `project`, `latitude`,
  `longitude`, extra columns).
- `recordings` — adds one nullable column vs. other backends:
  `score_chunk_path` (`id`, `filename`, `datetime`, `deployment_id`,
  `score_chunk_path`, extra columns). `score_chunk_path` is NULL until the
  recording's scores have been written, then points at the chunk Parquet
  file that contains its rows — see Q9.
- `detections` — new (see above); the id-bearing, reviewable unit that
  replaces `windows`.
- `annotations` — unchanged (human labels, keyed by `recording_id` +
  offsets, same as other backends).

**Score store (Parquet, one or more chunk files per deployment — see Q9):**
- Columns: `recording_id` (or `file` path, matching whichever `recordings`
  join key is chosen), `start_time`, `end_time`, and one column per class
  (`float16` scores). No synthetic row id.
- File-level key-value metadata (per the strategy doc): `model_id`/version,
  checkpoint hash, `clip_duration_s`, `clip_step_s`, `output_type`,
  `prediction_dtype`, `class_vocabulary_version`, `preprocessing_version`,
  `creation_timestamp`. This mirrors what `hoplite_metadata` already does at
  the whole-DB level (e.g. `usearch_config`), just scoped per Parquet file.
- Queried via DuckDB's `parquet_scan()`/`read_parquet()`, either globbed
  across all of a deployment's chunk files or resolved directly from
  `recordings.score_chunk_path` when the query is already scoped to specific
  recordings, joined against the native `recordings`/`deployments` tables in
  the same connection (no ATTACH needed, since everything is already
  DuckDB-native).

## Key design questions

### Q1: SQLite vs. DuckDB for the relational tables? — RESOLVED: pure DuckDB
Considered keeping SQLite for the relational tables (reusing
`sqlite_usearch_impl`'s SQL almost verbatim) with DuckDB used only
transiently to read Parquet via `ATTACH ... (TYPE sqlite)`. Rejected because
it requires two client libraries/connections in the same class, and because
the real concurrency need (below) is best served by a single owning
connection rather than two engines with different locking models.

DuckDB's own concurrency model doesn't allow a read-write connection to
coexist with any other connection (read or write) on the same file across
processes — unlike SQLite's WAL mode. Given the actual usage pattern (a user
reading/writing annotations while a background task writes new score
Parquets and their metadata) is genuinely concurrent-write, not just
concurrent-read, the resolution is: **a single writer-owning
process/connection**, with any concurrent producers (UI, background scoring
job) routing their writes through that one process. This coordination is a
**user-side concern** (their pipeline/UI architecture), not something
Hoplite's DB class needs to implement — the same way `SQLiteUSearchDB`
doesn't orchestrate multi-process access either, it just picks safe on-disk
semantics (WAL) and leaves coordination to the caller. Hoplite's job here is
just to pick DuckDB as the single, consistent engine and document the
single-writer-connection assumption; it does not need to build a queuing/RPC
layer.

> response: complete and pure duckdb + parquet implementation, no sqlite.


### Q2: Score insertion API — RESOLVED: batch-only
`insert_window()` (per-row, single-embedding) has no natural analogue here
since there is no `windows` table. Scores are written exclusively via a
batch API, e.g. `insert_deployment_scores(deployment_id, scores_df,
class_names, parquet_metadata, on_existing='skip')`, which writes one new
chunk Parquet file per call (see Q9 — a deployment may be written in one
call or several, at the caller's discretion). This matches how scoring
actually happens (one inference pass over a whole deployment, or several
passes if the caller chunks it for memory reasons) and avoids ever doing
row-by-row Parquet writes (extremely slow — Parquet is not designed for
single-row appends). Per-detection insertion (`insert_detection`/
`insert_detections_batch`, into the small `detections` table) is separate
and cheap either way, since that table is a normal DuckDB table, not
Parquet.

> a is good, but we will primarily or exclusively use batch insertion methods.

### Q3: How to key/locate a score row? — RESOLVED: no per-row id, key by (recording, offsets)
Superseded by the "Rethinking `windows`" decision above: there is no
per-clip table and no synthetic id minted for score rows at all. A score row
is addressed purely by `(recording_id, start_time, end_time)`, which already
uniquely locates it within its deployment's Parquet file — DuckDB joins
against `recordings` handle everything that needs recording-level metadata.

The only place a stable integer id is still needed is for **detections**
(sparse, reviewable, potentially annotated) — those get a real autoincrement
`id` in the new `detections` table, playing the role `windows.id` plays in
other backends, but scoped to the much smaller set of thresholded hits
rather than every clip.

### Q4: What replaces `get_embedding()` / `get_embeddings_batch()` / `search()` / window methods? — RESOLVED
`get_embedding`, `get_embeddings_batch`, and `search` all raise
`NotImplementedError` — there is no embedding vector and no ANN search
concept in this backend. Likewise `insert_window`, `get_window`,
`remove_window`, and `get_window_annotations` raise `NotImplementedError`,
since there is no `windows` table (see "Rethinking `windows`" above).

Separate, explicitly-named methods are added for the real access patterns:
- `insert_deployment_scores(...)` / whole-file Parquet write (Q2). Shipped
  as `insert_deployment_scores`, not `insert_scores_batch` as originally
  named here.
- `get_class_scores(class_name, deployments_filter=..., threshold=...)` —
  the "top scores for one class across many files" query from the notebook,
  using DuckDB column projection + row-group pruning against the Parquet
  scan(s), optionally joined with `recordings`/`deployments`. **Superseded
  by the richer `select_scores(...)`** actually shipped — see
  "Implementation status" at the top of this doc.
- `get_scores_batch(deployment_id, recording_ids, start_times, end_times)` —
  shipped as planned: a batch score-vector fetch by explicit `(recording,
  offsets)` keys, for cases that need the full class vector at specific
  clips (e.g. context around a detection).
- `insert_detection(s)`, `get_detection`, `remove_detection`,
  `get_detection_annotations` — the `detections`-table CRUD that plays the
  annotation-adjacent role `windows` played in other backends (see below).
  Shipped as planned, plus `select_detections(...)` and
  `add_detections_from_threshold(...)` for bulk detection queries/population.

> these methods will return not implemented errors, and we will implement separate methods for score querying and retrieval.

### Q5: Schema evolution — what happens when the class vocabulary changes?
Wide Parquet-per-deployment means class columns are fixed per file at write
time. Two realistic scenarios need an answer:
- A new model run adds/removes classes for a deployment that already has a
  Parquet file (e.g. re-scoring with an updated class list, or a
  model upgrade). Do we overwrite the file, version it
  (`{deployment}_{model}_{version}_{timestamp}.parquet`, as the notebook
  does), or require an explicit migration step?
- Two deployments scored with *different* class vocabularies (common when
  projects run different models over time) — a `parquet_scan('*.parquet')`
  across files with different columns will hit DuckDB's schema-unification
  behavior (missing columns become NULL, which is probably fine, but worth
  being explicit about) rather than erroring.

  **My recommendation**: version filenames by (model_id, model_version,
  creation_timestamp) as in the notebook rather than overwriting in place,
  and keep a small index (could just be a DuckDB/SQLite table, e.g.
  `score_files(deployment_id, model_id, model_version, class_vocabulary_version,
  path, created_at)`) so callers/queries can select "the latest scores for
  deployment X" without scanning the filesystem. **Do you want to support
  multiple model runs/versions coexisting per deployment (needs this
  index), or is "one current Parquet file per deployment, overwritten on
  rescore" sufficient for now?**

  > we will treat the class list as immutable for the entire database, as the database
  must only ever contain scores from a single versioned classifier. 

### Q6: Mutability — `remove_deployment` / `remove_recording` — RESOLVED
Orphaned Parquet rows were rejected: leaving deleted recordings' scores
behind in Parquet risks accidentally surfacing them again in a later query.
Since Parquet only supports whole-file rewrites (not row-level delete), and
every chunk file's rows belong to exactly one deployment (Q9):

- `remove_recording()` → `NotImplementedError`. Deleting a single
  recording's rows would require rewriting the whole chunk file it lives in
  (located via `recordings.score_chunk_path`); not supported.
- `remove_deployment()` → deletes the deployment's metadata rows
  (`deployments`, cascading to `recordings`/`annotations`/`detections` as
  today) **and** deletes all of the deployment's chunk Parquet files
  outright (`SELECT DISTINCT score_chunk_path FROM recordings WHERE
  deployment_id = ?`, then unlink each). This is exact (no orphaning) and
  cheap (file deletes, not rewrites), since a chunk never spans more than
  one deployment.

### Q7: Concurrency / write model — RESOLVED
A single Parquet file has a single writer (whoever computed that
deployment's scores), so no in-library multi-writer support is needed for
Parquet itself. Combined with Q1's resolution, the whole DB (metadata +
Parquet) is designed around one owning connection/process; any coordination
across concurrent producers (UI, background scoring job) is the caller's
responsibility, not something built into this module.

### Q8: Float precision and NaN/missing-class handling — RESOLVED
`output_type` and similar fields are opaque, caller-supplied metadata with no
backend validation/enforcement. The missing-column question is moot: the
class vocabulary is fixed for the life of the database (Q5), so every
deployment's Parquet file has the same columns and there is no
union-of-schemas scenario to handle.

### Q9: Chunked score files and `recordings.score_chunk_path` — RESOLVED, then not built
**Not implemented.** The design below was the plan, but the shipped
`insert_deployment_scores()` is simpler: exactly one Parquet file per
deployment, written as a whole-file replacement, no `score_chunk_path`
column, no chunk numbering, no `on_existing='skip'` dedup logic (only
`handle_existing='error'|'overwrite'`, where `'overwrite'` replaces the
entire file with exactly the batch passed in). See "Implementation status"
at the top of this doc. Left below for the reasoning trail in case a real
deployment does turn out too large to hold in memory for one write, which
would make this design worth revisiting.

**Now being built as an opt-in extension** — see
`PARQUET_CHUNKING_IMPLEMENTATION_PLAN.md` for the concrete, code-level
implementation plan (exact call sites, schema migration, test additions),
written against the actual shipped `parquet_scores_impl.py` rather than
this speculative design.

For most deployments, holding the whole score table in memory before one
Parquet write is fine (MB-scale; even 100k windows × 15k classes
unthresholded is ~6GB, which is acceptable as a rare worst case). But a
caller with a very large deployment or a memory-constrained machine should
be able to split the write into several smaller calls instead. Resolution:
make the **chunk file**, not the deployment, the unit a single
`insert_deployment_scores()` call produces — a deployment ends up backed by
one or more chunk Parquet files, and the caller controls chunking only by
choosing how many recordings' worth of scores to pass to each call, not by
naming or numbering chunks themselves.

- **One call = one new chunk file.** Each `insert_deployment_scores()` call
  writes exactly one new Parquet file (internally named/numbered, e.g.
  `{deployment_id}_{n}.parquet` with `n` auto-assigned by listing existing
  chunks for that deployment); it never appends to or rewrites an existing
  chunk as part of a normal write. A caller who scores a whole deployment in
  one call gets exactly the original one-file-per-deployment behavior; a
  caller who calls it once per recording (or per N recordings) gets many
  small chunk files instead, bounding peak memory to whatever they choose to
  pass per call.
- **`recordings.score_chunk_path`** (new nullable column, see "Proposed
  structure" above) records which chunk file holds a given recording's
  scores, NULL until first written. This is what makes a recording's scores
  locatable in O(1) regardless of how many chunks a deployment has, and is
  the cheap check that powers the next point.
- **Handling a recording that's already been scored.** Before writing a new
  chunk, look up `score_chunk_path` for every recording in the incoming
  batch. Given the database already commits to one fixed, versioned
  classifier for its whole lifetime (Q5/Q8) — so two score rows for the same
  `(recording_id, start_time, end_time)` should never legitimately disagree
  — the default is to treat a duplicate as redundant, not conflicting:
  - `on_existing='skip'` (default) — drop any recording that already has a
    `score_chunk_path` from the batch before writing; cheap (one column
    lookup per recording, touches only the new chunk file). Safe because a
    caller who only ever calls this once per deployment never has any
    recording with a pre-existing `score_chunk_path`, so this check is a
    no-op for that (common) usage.
  - `on_existing='error'` — same lookup, but raise if any recording in the
    batch already has a chunk path, for callers who want to assert "these
    are all new" rather than silently drop rows.
  - `on_existing='overwrite'` — for correcting a bad run, not routine use.
    Locate the old chunk file(s) referenced by the affected recordings
    (possibly several, if the batch spans recordings written in different
    prior calls), rewrite each minus the stale rows, write the new chunk,
    and update `score_chunk_path` for every moved recording. Unlike `skip`/
    `error`, cost here is *not* bounded by the new batch's size — it's
    bounded by however fragmented the affected recordings' prior chunks are
    — so this mode should be documented as the expensive path.
- **Write ordering / crash safety.** A chunk file write (filesystem) and the
  `recordings.score_chunk_path` update (DuckDB) are not one atomic
  transaction. Write the new chunk file first (to a temp path, then atomic
  rename into place), and only update `recordings` afterward. If the
  process dies between the two, the result is an orphaned-but-harmless
  Parquet file — never a `recordings` row pointing at a chunk file that
  doesn't exist, which would break reads.
- **Query side.** `parquet_scan()` over a deployment now globs its chunk
  files (`{deployment_id}_*.parquet`) or, when a query is already scoped to
  specific recordings, resolves `score_chunk_path` directly per recording
  and reads only the relevant chunk(s) — a pruning win over globbing when
  the caller already knows which recordings it wants.
- **Why chunk boundaries stay deployment-scoped rather than fully
  arbitrary** (considered and rejected): keeping every chunk file's rows
  drawn from a single deployment is what keeps `remove_deployment()` a plain
  file-delete (Q6) instead of a rewrite-minus-rows operation, and what lets
  deployment-filtered queries skip whole files at the DuckDB planning stage
  instead of opening every file and filtering rows after the join. Both
  properties would be lost if a chunk could mix recordings from different
  deployments.

## Suggested next steps once the above are answered — mostly DONE

1. ~~Design the DuckDB DDL and filter-dict → SQL compiler~~ — DONE, see
   `deployments`/`recordings`/`annotations`/`detections`/`hoplite_metadata`
   DDL in `_setup_tables()` and the filter compiler
   (`format_sql_where_conditions`) in `parquet_scores_impl.py`.
2. ~~Implement `parquet_scores_impl.py`~~ — DONE, but simpler than planned:
   one Parquet file per deployment (no chunking/`score_chunk_path`; see
   "Implementation status" above), written whole-file by
   `insert_deployment_scores()`, read via `select_scores()`/
   `get_scores_batch()` (not the originally-named `get_class_scores`).
3. ~~Implement `detections` CRUD~~ — DONE: `insert_detection(s)`,
   `get_detection`, `remove_detection`, `get_detection_annotations`,
   `select_detections()`, and `add_detections_from_threshold()` all exist
   as planned.
4. **Not done.** `db/tests/test_utils.py`'s `DB_TYPES`/`make_db()` (the
   shared conformance suite run by `hoplite_test.py`/`interface_test.py`
   across backends) has no `parquet_scores` entry — `ParquetScoresDB` is
   exercised only by its own standalone
   `db/tests/parquet_scores_impl_test.py`, not the cross-backend suite.
   Worth doing if/when this backend needs to guarantee interface parity
   with the other backends on the parts that do apply
   (deployments/recordings/annotations/metadata), with the window/embedding
   tests marked not-applicable (`NotImplementedError` expected) for this
   backend.
5. ~~Add scores- and detections-specific tests~~ — DONE:
   `db/tests/parquet_scores_impl_test.py` (~1,500 lines) covers deployment/
   recording/annotation/detection CRUD, extra columns, duplicate handling,
   `remove_deployment()`'s exact-delete behavior, and more. Separately,
   `parquet_scores_stress_test.ipynb` and
   `parquet_scores_class_scaling_test.ipynb` cover realistic-scale
   performance characteristics (not correctness) for `select_scores()`
   and friends — see "Implementation status" above for the class-scaling
   finding.
