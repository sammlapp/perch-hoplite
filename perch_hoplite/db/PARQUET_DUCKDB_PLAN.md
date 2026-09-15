# Plan: Parquet + DuckDB/SQLite backend for wide per-window class scores

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
- `recordings` — unchanged (`id`, `filename`, `datetime`, `deployment_id`,
  extra columns).
- `detections` — new (see above); the id-bearing, reviewable unit that
  replaces `windows`.
- `annotations` — unchanged (human labels, keyed by `recording_id` +
  offsets, same as other backends).

**Score store (Parquet, one file per deployment):**
- Columns: `recording_id` (or `file` path, matching whichever `recordings`
  join key is chosen), `start_time`, `end_time`, and one column per class
  (`float16` scores). No synthetic row id.
- File-level key-value metadata (per the strategy doc): `model_id`/version,
  checkpoint hash, `clip_duration_s`, `clip_step_s`, `output_type`,
  `prediction_dtype`, `class_vocabulary_version`, `preprocessing_version`,
  `creation_timestamp`. This mirrors what `hoplite_metadata` already does at
  the whole-DB level (e.g. `usearch_config`), just scoped per Parquet file.
- Queried via DuckDB's `parquet_scan()`/`read_parquet()`, joined directly
  against the native `recordings`/`deployments` tables in the same
  connection (no ATTACH needed, since everything is already DuckDB-native).

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
batch API, e.g. `insert_scores_batch(deployment_id, scores_df, class_names,
parquet_metadata)`, which writes the deployment's Parquet file in one shot.
This matches how scoring actually happens (one inference pass over a whole
deployment) and avoids ever doing row-by-row Parquet writes (extremely slow
— Parquet is not designed for single-row appends). Per-detection insertion
(`insert_detection`/`insert_detections_batch`, into the small `detections`
table) is separate and cheap either way, since that table is a normal
DuckDB table, not Parquet.

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
- `insert_scores_batch(...)` / bulk Parquet write (Q2).
- `get_class_scores(class_name, deployments_filter=..., threshold=...)` —
  the "top scores for one class across many files" query from the notebook,
  using DuckDB column projection + row-group pruning against the Parquet
  scan(s), optionally joined with `recordings`/`deployments`.
- `get_scores_batch(recording_ids, start_times, end_times)` or similar — a
  batch score-vector fetch by explicit `(recording, offsets)` keys, for
  cases that need the full class vector at specific clips (e.g. context
  around a detection).
- `insert_detection(s)`, `get_detection`, `remove_detection`,
  `get_detection_annotations` — the `detections`-table CRUD that plays the
  annotation-adjacent role `windows` played in other backends (see below).

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
there is one Parquet file per deployment:

- `remove_recording()` → `NotImplementedError`. Deleting a single
  recording's rows would require rewriting its whole deployment's Parquet
  file; not supported.
- `remove_deployment()` → deletes the deployment's metadata rows
  (`deployments`, cascading to `recordings`/`annotations`/`detections` as
  today) **and** deletes the deployment's entire Parquet file outright. This
  is exact (no orphaning) and cheap (file delete, not rewrite), since
  deployment is exactly the granularity at which Parquet files are split.

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

## Suggested next steps once the above are answered
1. Design the DuckDB DDL and filter-dict → SQL compiler for
   `deployments`/`recordings`/`annotations`/`detections`/`hoplite_metadata`,
   using `sqlite_usearch_impl.py` as a reference for shape/semantics (not
   for reusable code, since the dialect differs).
2. Implement `parquet_scores_impl.py`: a single DuckDB connection owning the
   above tables, plus a Parquet writer/reader pair for the per-deployment
   score files (bulk write in `insert_scores_batch`, DuckDB `parquet_scan`
   reads for `get_class_scores`/`get_scores_batch`).
3. Implement `detections` CRUD (`insert_detection(s)`, `get_detection`,
   `remove_detection`, `get_detection_annotations`) and the
   threshold-scan helper (`add_detections_from_threshold` or similar) that
   populates it from a Parquet scan.
4. Add a `parquet_scores` entry to `db/tests/test_utils.py`'s `DB_TYPES` /
   `make_db()` so the existing conformance suite
   (`hoplite_test.py`/`interface_test.py`) runs against it for the
   deployments/recordings/annotations/metadata parts of the interface,
   marking the window/embedding-specific tests as not-applicable
   (`NotImplementedError` expected) for this backend.
5. Add scores- and detections-specific tests mirroring the notebook's "top
   score per class across deployments" query, the metadata-provenance
   round-trip, and `remove_deployment()`'s exact-delete behavior.
