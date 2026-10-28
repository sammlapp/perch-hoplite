# `ParquetScoresDB`: design and implementation

`perch_hoplite/db/parquet_scores_impl.py` implements `HopliteDBInterface` for
**per-window classifier scores** (a vector of class scores per window,
typically 10–500 classes and up to ~10k) and for **detections** (sparse
single-class hits above a threshold). Other backends (`in_mem_impl`,
`sqlite_usearch_impl`) store one embedding per window for nearest-neighbor
search. This one serves column-wise reads instead ("all/top scores for class
X across many files, filtered by deployment/recording metadata") and does no
vector search.

Install with `pip install perch-hoplite[parquet]` (`duckdb>=1.0,<2.0`,
`pyarrow>=17.0`). For a hands-on walkthrough, see
`parquet_scores_api_tour.ipynb`.

## Storage layout

```
<db_path>/
  hoplite.duckdb                     # all relational tables
  scores/
    deployment_<id>_<n>.parquet      # one or more score chunk files per deployment
```

### DuckDB tables

All of these are DuckDB tables in one `.duckdb` file. SQLite is not used.

| table | columns (built-in) | notes |
|---|---|---|
| `hoplite_metadata` | `key`, `value` (JSON) | holds `class_labels` |
| `deployments` | `id`, `name`, `project`, `latitude`, `longitude` | `UNIQUE(name, project)` |
| `recordings` | `id`, `filename`, `datetime` (`TIMESTAMPTZ`), `deployment_id`, `score_chunk_path` | `UNIQUE(filename, deployment_id)` |
| `annotations` | `id`, `recording_id`, `start_time`, `end_time`, `label`, `label_type`, `provenance` | same as other backends |
| `detections` | `id`, `recording_id`, `start_time`, `end_time`, `class_name`, `score` | `UNIQUE(recording_id, start_time, end_time, class_name)` |

Indexes: `recordings(deployment_id)`, `annotations(recording_id)`,
`detections(recording_id)`, `detections(class_name)`. Every table except
`hoplite_metadata` accepts extra columns through `add_extra_table_column()`.
Supported types: `int`, `float`, `str`, `bytes`, `datetime`, `list`.
`insert_*` methods add extra columns automatically from `**kwargs`.

### Score chunk files (Parquet)

- Columns: `recording_id` (int64), `start_time` (float64), `end_time`
  (float64), then one **float16** column per class, named exactly as the class
  label and ordered as in `classes`.
- There is no synthetic row id. A score row is identified by
  `(recording_id, start_time, end_time)`.
- Each chunk holds rows from **exactly one deployment**. A deployment can have
  any number of chunks.
- `recordings.score_chunk_path` is the source of truth for where a
  recording's scores are. It is NULL until scores are written. All of a
  recording's rows are in a single chunk.

## Design decisions

1. **DuckDB only.** One engine handles both metadata mutations and reads that
   join Parquet files, through a single connection. We considered and rejected
   SQLite for metadata with DuckDB attached for Parquet reads: it needs two
   client libraries and two locking models.

2. **Single writer.** A read-write DuckDB connection is exclusive and cannot
   coexist with any other connection to the same file. Callers with several
   producers (e.g. an annotation UI and a background scoring job) must route
   all writes through one owning process. The module does not provide any
   queuing for this. In-process threads should use `thread_split()`, which
   returns a `cursor()` on the same connection (DuckDB refuses a second
   `connect()` to the same file) and sets `TimeZone='UTC'` again on it.

3. **No `windows` table.** At realistic scale (1M windows), a per-window table
   would add 1M rows that only hold bookkeeping. `(recording_id, start_time,
   end_time)` already identifies a score row. These methods raise
   `NotImplementedError`: `insert_window`, `get_window`, `remove_window`,
   `get_window_annotations`, `get_all_windows`, `match_window_ids`,
   `get_embedding`, `count_embeddings`, and `search`.

4. **`detections` takes the place of `windows` as the row that has an id.**
   Thresholding scores yields a small set of `(recording, offsets, class,
   score)` hits. These hits get ids and can be reviewed. Annotations stay
   independent and are matched to detections by time overlap.

5. **The class vocabulary is fixed for the life of the database.** A database
   holds scores from one versioned classifier. `class_labels` is required at
   first `create()`, saved in `hoplite_metadata`, and validated on every
   later open. Every chunk therefore has an identical schema, with no schema
   unification and no missing-class columns.

6. **Scores are written only in batches.** Parquet does not support row
   appends, so `insert_deployment_scores()` is the only way to write scores.

7. **The chunk is the write unit, and chunk boundaries stay within a
   deployment.** One `insert_deployment_scores()` call writes at most one new
   chunk. The caller controls chunking only through how they batch calls. A
   caller who writes a whole deployment in one call gets one file per
   deployment. Keeping each chunk within one deployment has two benefits:
   `remove_deployment()` is a plain file delete rather than a rewrite, and
   deployment-scoped queries never open other deployments' files.

8. **Deletion is exact, never orphaned.** `remove_deployment()` deletes the
   deployment's rows in `deployments`, `recordings`, `annotations` and
   `detections`, then unlinks all of its chunk files. DuckDB has no
   `ON DELETE CASCADE`, so the cascade is written in application code.
   `remove_recording()` raises `NotImplementedError`, because removing one
   recording would mean rewriting its chunk.

9. **Datetimes must be timezone-aware.** Naive datetimes are rejected
   everywhere through `normalize_sql_value` / `require_timezone_aware`. The
   session pins `TimeZone='UTC'`, so `TIMESTAMPTZ` handling does not depend
   on the host.

10. **Upserts keep values the caller omitted.** For deployments, recordings
    and detections, a duplicate key updates the row with
    `COALESCE(excluded.col, existing.col)`. Passing `None` or leaving a field
    out therefore never blanks a stored value.

## API

### Lifecycle and metadata

- `ParquetScoresDB.create(db_path, class_labels=None, readonly=False)`
- `classes` (a tuple of class labels), `get_embedding_dim()` (equal to
  `len(classes)`), `get_embedding_dtype()` (`np.float16`)
- `commit()`, `rollback()`, `thread_split()`
- `insert_metadata` / `get_metadata` / `remove_metadata`
- `add_extra_table_column`, `get_extra_table_columns`

### Deployments, recordings, annotations

These follow the standard interface: `insert_deployment`, `get_deployment`,
`remove_deployment`, `get_all_deployments`, `get_all_projects`,
`insert_recording`, `get_recording`, `get_all_recordings`,
`count_recordings`, `insert_annotation`, `get_annotation`,
`remove_annotation`, `get_all_annotations`, `get_all_labels`,
`count_each_label`.

### Writing scores

```python
db.insert_deployment_scores(
    deployment_id, recording_ids, start_times, end_times,
    scores,                                   # [N, len(db.classes)]
    handle_existing_recordings='error',       # | 'skip' | 'overwrite'
)
```

`handle_existing_recordings` is checked **per recording**, not per row, using
`score_chunk_path`:

- `'error'` (default): raises before writing anything if any recording in the
  batch already has scores. A caller that writes each deployment once sees
  the same behavior as a single file per deployment.
- `'skip'`: drops recordings that already have scores and writes the rest to
  a new chunk.
- `'overwrite'`: for recordings that already have scores, replaces **all**
  their existing rows with the rows in this batch. The update happens in place
  in the chunks they already live in, so `score_chunk_path` does not change.
  Only recordings that had no scores go to a new chunk. Cost scales with the
  number of existing chunks involved, because each affected chunk is read in
  full and rewritten. This path is meant for fixing a bad run, not for routine
  use.

If the batch is empty after filtering, nothing is written: no zero-row chunk
file and no `recordings` update.

### Reading scores

| method | returns | use |
|---|---|---|
| `select_scores(class_name, selection, n, ...)` | DataFrame | top / random / percentile-binned rows for one class, optionally stratified |
| `get_scores(deployments_filter, recordings_filter, start_date, end_date)` | `ScoreBatch` | every window that matches the filters, with full score vectors |
| `get_scores_batch(deployment_id, recording_ids, start_times, end_times)` | `ndarray [N, n_classes]` | full vectors for specific windows (e.g. context around a detection); raises `KeyError` for a missing key |
| `sampling_effort(by_deployment, date_bins, ...)` | DataFrame | window counts per stratum, **including zero-count strata** |
| `count_windows(deployment_id=None)` | int | total rows, read from Parquet footers only |

`select_scores` arguments:
- `selection='top'|'random'|'binned'`, with `n` rows per stratum (per bin for
  `'binned'`). `'binned'` takes `bin_edges` as percentiles in `[0, 1]`; cut
  points come from `quantile_cont` within each stratum.
- Stratification options can be combined: `by_deployment` (shorthand for
  `by_deployment_fields=['id']`), `by_deployment_fields` (any `deployments`
  columns, including extra columns; each one also becomes an output column),
  and `date_bins` (`'day'`, `'week'`, or explicit `(tz-aware start, label)`
  pairs assigned with an `ASOF JOIN`).
- Filters: `deployments_filter`, `recordings_filter`, `min_score`/`max_score`,
  and `start_date`/`end_date`.
- `random_seed` makes `'random'` sampling reproducible.

`ScoreBatch` is struct-of-arrays: `deployment_ids`, `recording_ids`,
`start_times`, `end_times`, `scores [N, n_classes]`, `class_labels`. It
supports `len`, `batch[i]` and `to_pandas()`, which gives one column per
class.

### Detections

- `insert_detection(recording_id, offsets, class_name, score,
  handle_duplicates='overwrite'|'skip'|'error', **extras)`,
  `insert_detections_batch`, `get_detection`, `remove_detection`,
  `get_all_detections`, `count_detections`
- `get_detection_annotations(detection_id, label=None)`: returns annotations
  that overlap the detection in time.
- `add_detections_from_threshold(threshold, class_name=None,
  deployments_filter, recordings_filter, handle_duplicates, **extras)`: scans
  chunks for `score >= threshold`. If `class_name` is omitted, all classes are
  scanned in one `UNPIVOT` query, so a single window can produce several
  detections.
- `select_detections(...)`: same selection, stratification and filter options
  as `select_scores`, plus a `filter` on the detections table. Here
  `class_name` is optional.

### Helpers (module-level)

`add_deployment_metadata(db, df, foreign_key='deployment_id')` and
`add_recording_metadata(db, df, foreign_key='recording_id')` left-join
deployment or recording columns, including extra columns, onto any DataFrame
(from `select_scores`, `ScoreBatch.to_pandas()`, etc.). The joined metadata
columns are placed first.

### Filters

Filters use the same `ConfigDict` operators as the other backends: `eq`,
`neq`, `lt`, `lte`, `gt`, `gte`, `isin`, `notin`, `range`, `approx`.
`offsets` is accepted as a pseudo-column, with `approx` only, on annotations
and detections. Unknown column names raise a clear `ValueError` up front.
**Common mistake:** `deployments_filter` uses the deployment's own primary
key `id` (e.g. `eq=dict(id=3)`), not `deployment_id`.

## Implementation details

### Chunk naming and discovery
- `_next_chunk_path(deployment_id)` lists `scores/` with a glob for
  `deployment_<id>_*.parquet` and uses the highest numeric suffix + 1. It
  checks the filesystem rather than `recordings`, so a chunk left on disk
  without a matching `recordings` row (see the next section) can never be
  overwritten.
- `deployment_chunk_paths(deployment_id)` returns `SELECT DISTINCT
  score_chunk_path FROM recordings WHERE deployment_id = ?`. Reads use only
  chunks that `recordings` points to.

### Write ordering and crash safety
1. Write the chunk to `<chunk>.parquet.tmp` with `pyarrow.parquet`, then
   `replace()` it into place. The replace is atomic, so readers never see a
   partial file.
2. Only then run `UPDATE recordings SET score_chunk_path = ...`.

If the process crashes between the two steps, it leaves an orphaned chunk
file that nothing reads, but never a `recordings` row pointing at a missing
file. In-place overwrites use the same tmp + replace approach for each
affected chunk.

### Float16 and the Arrow import path
Writing goes through `pyarrow.parquet` rather than DuckDB. DuckDB's Arrow
import fails on half-float columns. In-place overwrite reads with
`pq.read_table` rather than DuckDB, because DuckDB's Arrow export widens
float16 to float32, which would break `concat_tables` against the new float16
rows. DuckDB's `read_parquet` reads float16 without problems, so all queries
use it.

### Query construction
- Every score read builds one `UNION ALL` branch **per deployment**. Each
  branch is `read_parquet([chunk1, chunk2, ...])` over that deployment's
  chunks and attaches the constant `deployment_id`. The four variants:
  - `_scores_union_query`: one class column, aliased `score`.
  - `_scores_union_query_all_classes`: `UNPIVOT` to rows of
    `(window, class_name, score)`.
  - `_windows_union_query`: key columns only, for counting.
  - `_wide_scores_union_query`: all class columns.
- The union is joined to `recordings` and `deployments` for metadata filters
  and dates. Recording metadata is never stored in Parquet.
- `get_scores_batch` reads only the chunks that hold the requested
  recordings, looked up through `score_chunk_path`. It loads the request keys
  into a temp table and `LEFT JOIN`s on `recording_id` and offsets (with
  ±1e-6 tolerance), so missing keys are detected.
- Class names often contain spaces or hyphens. They are checked against
  `classes`, then double-quoted with `"` escaped (`quote_sql_identifier`).
  Table and column names in the relational schema must match
  `^[a-zA-Z_][a-zA-Z0-9_]*$`.
- Seeded random selection orders by `hash(row key || seed)` instead of
  `setseed()` + `random()`. Parallel scans consume `random()` in a
  nondeterministic order.
- Temp tables (`_select_scores_buckets`, `_sampling_effort_grid`,
  `_score_keys`) are dropped in `finally` blocks.

### DuckDB quirks worked around
- **FK-referenced upsert:** DuckDB rejects `INSERT ... ON CONFLICT DO UPDATE`
  on a row that a foreign key references, even when the update changes
  nothing. `_upsert_row` catches this case and falls back to a lookup + plain
  `UPDATE`.
- **No `ON DELETE CASCADE`:** cascades are written by hand in
  `remove_deployment`.

## Performance notes

- Selecting one class reads only that column (column projection) from each
  chunk. In `parquet_scores_class_scaling_test.ipynb`, the cost of growing
  from 1 to 1,000 total classes is ~10× on tiny files (~13 ms → ~130 ms;
  parsing the Parquet footer and schema dominates) and only ~2× at realistic
  sizes (100k windows/deployment). Class count is not a practical concern at
  real deployment sizes.
- `parquet_scores_stress_test.ipynb` covers performance at realistic scale.
- `count_windows` reads only Parquet footers.
- `add_detections_from_threshold` scans in one query but inserts detections
  one row at a time through `insert_detection`. That is fine for thousands of
  hits and slow for millions.

## Known limitations

- **`score_chunk_path` stores the full path** (`db_path/scores/...`, as passed
  to `create()`). If you move or rename the database directory, or open it
  through a different relative path, reads will point at stale paths. Storing
  paths relative to `scores_dir` would fix this.
- **No schema migration.** `_setup_tables()` returns immediately if
  `recordings` already exists. A `.duckdb` file created before
  `score_chunk_path` was added will not gain the column. Files named in the
  older format, `deployment_<id>.parquet` (no suffix), are not read. Such
  databases must be recreated.
- Each class vocabulary needs its own database. Mixing models in one
  database is not supported.
- `remove_recording` is not supported.
- `ParquetScoresDB` is not registered in `db/tests/test_utils.py`'s
  `DB_TYPES`, because the shared conformance suite assumes a windows table and
  embeddings. It is tested on its own in
  `db/tests/parquet_scores_impl_test.py`, which includes multi-chunk tests
  covering `skip`/`error`/`overwrite`, overwrites spanning several chunks,
  and removing a deployment that has several chunks.
