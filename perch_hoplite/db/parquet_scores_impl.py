# coding=utf-8
# Copyright 2026 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DuckDB + Parquet database implementation for wide per-window class scores
and/or discrete event detections.

Unlike `sqlite_usearch_impl.SQLiteUSearchDB` (built around one embedding
vector per window, indexed for approximate nearest-neighbor search), this
backend handles classification outputs in two formats:
1. model outputs that are a wide table: a *vector of class scores per
window* (e.g. 10-500 classes, up to ~10k), where the dominant access pattern
is columnar ("give me every/top scores for class X across many files") rather
than vector similarity search.
2. sparse, thresholded *detections* (single-class hits) with time bounds

Design (see `PARQUET_DUCKDB_PLAN.md` for the full design discussion):
  - All relational metadata (`hoplite_metadata`, `deployments`, `recordings`,
    `annotations`, `detections`) lives in native DuckDB tables in a single
    `.duckdb` file, read and written through a single DuckDB connection.
    There is no SQLite involved anywhere in this backend.
  - There is no `windows` table. A "window" (a scored clip) is addressed
    purely by `(recording_id, start_time, end_time)`, which already uniquely
    locates its row in its deployment's score chunk file(s); no synthetic
    per-clip id is minted. `insert_window`, `get_window`, `remove_window`,
    and `get_window_annotations` all raise `NotImplementedError`.
  - The wide class-score matrix itself lives in one or more Parquet "chunk"
    files per deployment (not in DuckDB tables), written in bulk (one new
    chunk file per `insert_deployment_scores()` call -- see that method and
    PARQUET_DUCKDB_PLAN.md, Q9) and read via DuckDB's
    `parquet_scan`/`read_parquet` over all of a deployment's chunks at
    once. `recordings.score_chunk_path` records which chunk file holds a
    given recording's scores.
  - `detections` is a new, small, id-bearing table for sparse
    score-thresholded single-class hits (see `datatypes.Detection`) -- this
    plays the role `windows` plays in other backends as the reviewable unit
    that annotations can be attached to, but at detection-count scale rather
    than clip-count scale.
  - There is no embedding vector and no ANN search in this backend:
    `get_embedding`, `get_embeddings_batch`, and `search` all raise
    `NotImplementedError`. Score retrieval goes through new, explicitly
    named methods instead (see the score-query section of this module).
  - The class vocabulary is fixed for the lifetime of a database (a database
    holds scores from a single versioned classifier).
  - DuckDB does not support `ON DELETE CASCADE` on foreign keys (unlike
    SQLite), so cascade deletes across
    deployments -> recordings -> annotations/detections are implemented
    explicitly in application code rather than relied upon from the schema.
  - DuckDB does not support concurrent read-write + any-other-connection
    access to the same database file the way SQLite's WAL mode does opening
    this database for read-write is exclusive. Callers that need concurrent
    writers (e.g. a UI editing annotations while a background job writes new
    deployments) must serialize those writes through a single owning
    connection/process themselves; this module does not implement any
    queuing or multi-writer coordination.
"""

from collections.abc import Sequence
import dataclasses
import datetime as dt
import functools
import json
import re
from typing import Any, Literal

from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface

# NOTE(parquet_scores): duckdb/pyarrow are optional extras (`pip install
# perch-hoplite[parquet]`); import lazily-ish at module scope since this
# module is never imported by code that doesn't want this backend.
import duckdb


HOPLITE_FILENAME = 'hoplite.duckdb'
SCORES_DIRNAME = 'scores'
CLASS_LABELS_KEY = 'class_labels'

PYTHON_TYPE_TO_SQL_TYPE = {
    int: 'INTEGER',
    float: 'DOUBLE',
    str: 'VARCHAR',
    bytes: 'BLOB',
    dt.datetime: 'TIMESTAMPTZ',
    list: 'DOUBLE[]',
}
SQL_TYPE_TO_PYTHON_TYPE = {
    'INTEGER': int,
    'BIGINT': int,
    'DOUBLE': float,
    'FLOAT': float,
    'VARCHAR': str,
    'BLOB': bytes,
    'TIMESTAMP WITH TIME ZONE': dt.datetime,
    'DOUBLE[]': list,
}

_METADATA_TABLES = (
    'deployments',
    'recordings',
    'annotations',
    'detections',
)

# The built-in (non-extra) columns of each metadata table, keyed by table
# name. Used both to distinguish extra columns from built-in ones (see
# `ParquetScoresDB._extra_table_columns`) and to validate filter column
# names up front with a clear error, rather than letting an unknown column
# surface as a raw `duckdb.BinderException` deep inside a query (see
# `ParquetScoresDB._validate_filter_columns`).
_DEFAULT_TABLE_COLUMNS = {
    'deployments': {'id', 'name', 'project', 'latitude', 'longitude'},
    'recordings': {
        'id',
        'filename',
        'datetime',
        'deployment_id',
        'score_chunk_path',
    },
    'annotations': {
        'id',
        'recording_id',
        'start_time',
        'end_time',
        'label',
        'label_type',
        'provenance',
    },
    'detections': {
        'id',
        'recording_id',
        'start_time',
        'end_time',
        'class_name',
        'score',
    },
}
# Pseudo-columns accepted by `format_sql_where_conditions` in addition to
# real columns of the named table (translated internally, e.g. `offsets` ->
# `start_time`/`end_time`); a filter using one of these should not be
# treated as an unknown-column error.
_FILTER_PSEUDO_COLUMNS = {
    'annotations': {'offsets'},
    'detections': {'offsets'},
}


def is_valid_sql_identifier(name: str) -> bool:
  """Check if a string is a valid and safe SQL identifier."""

  if not name or not isinstance(name, str):
    return False
  return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None


def quote_sql_identifier(name: str) -> str:
  """Double-quote a SQL identifier, escaping any embedded `"`.

  Unlike table/column names in the relational schema (validated via
  `is_valid_sql_identifier` and used unquoted), class names come from a
  caller-supplied vocabulary and commonly contain spaces/hyphens (e.g.
  "American Redstart") that are valid Parquet column names but not valid
  bare SQL identifiers. This applies standard SQL quoted-identifier
  escaping (doubling embedded `"`) so such names can still be referenced
  safely in a query.
  """
  return '"' + name.replace('"', '""') + '"'


def require_timezone_aware(value: dt.datetime | None, param_name: str) -> None:
  """Raise `ValueError` if `value` is a naive (no tzinfo) datetime.

  `recordings.datetime` and related date parameters are stored/compared as
  `TIMESTAMPTZ` (see `ParquetScoresDB.create()`), so accepting a naive
  datetime would force a decision about what timezone it means -- silently
  assuming UTC (or the host's local timezone, which is even worse) is
  exactly the kind of implicit behavior that causes confusing bugs later.
  Naive datetimes are rejected outright instead; callers must attach an
  explicit `tzinfo` (e.g. `dt.timezone.utc`).
  """
  if value is not None and value.tzinfo is None:
    raise ValueError(
        f'`{param_name}` must be a timezone-aware datetime (e.g. built with'
        ' `tzinfo=datetime.timezone.utc`); got a naive datetime'
        f' ({value!r}), which would require silently assuming a timezone.'
    )


def normalize_sql_value(value: Any) -> Any:
  """Normalize a python value to one of the types supported by DuckDB."""

  if (
      isinstance(value, list)
      or isinstance(value, tuple)
      or isinstance(value, np.ndarray)
  ):
    return [normalize_sql_value(v) for v in value]
  if isinstance(value, datatypes.LabelType):
    return value.value
  elif isinstance(value, dt.datetime):
    # Every inserted/filtered value flows through here (both the built-in
    # `recordings.datetime` column and any extra `dt.datetime`-typed column
    # added via `add_extra_table_column`), so this is the single choke
    # point for enforcing timezone-awareness everywhere, not just on the
    # named `datetime` parameter of `insert_recording`.
    require_timezone_aware(value, 'datetime value')
    return value
  elif isinstance(value, np.integer):
    return int(value)
  elif isinstance(value, np.floating):
    return float(value)
  return value


def format_sql_insert_values(
    **kwargs: Any,
) -> tuple[str, str, list[Any]]:
  """Build columns string, placeholders string and values list for INSERT.

  Args:
    **kwargs: Key-value pairs to pass to the SQL statement.

  Returns:
    A tuple of: a formatted columns string, a formatted placeholders string,
    and a list of corresponding values. Safe to use in SQL INSERT statements.
  """

  for key in kwargs:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  columns = list(kwargs.keys())
  placeholders = ['?'] * len(columns)
  values = normalize_sql_value(list(kwargs.values()))

  return f"({', '.join(columns)})", f"({', '.join(placeholders)})", values


def _is_fk_referenced_upsert_error(error: 'duckdb.Error') -> bool:
  """Whether `error` is DuckDB's FK-referenced-row upsert limitation.

  DuckDB refuses `INSERT ... ON CONFLICT ... DO UPDATE` against a row that
  is referenced by a foreign key in another table -- even when the update
  would be a no-op (e.g. re-inserting the same `(name, project)` deployment
  after it already has recordings). A plain `UPDATE ... WHERE id = ?`
  against the same row works fine; only the upsert form is rejected. See:
  https://duckdb.org/docs/sql/query_syntax/insert#on-conflict-clause
  """
  return isinstance(
      error, duckdb.ConstraintException
  ) and 'foreign key' in str(error).lower()


def _is_unique_constraint_error(error: 'duckdb.Error') -> bool:
  """Whether `error` is a plain unique-constraint violation.

  Used by `insert_detection(handle_duplicates='error')` to distinguish "a
  matching detection already exists" (which gets a specific, actionable
  `RuntimeError`) from other insert failures (e.g. a bad `recording_id`).
  """
  return isinstance(
      error, duckdb.ConstraintException
  ) and 'unique constraint' in str(error).lower()


def format_sql_update_on_conflict(table: str, *args: str) -> str:
  """Build the update part of ON CONFLICT clauses for INSERT statements.

  Updated columns use `COALESCE(excluded.col, table.col)` rather than a
  plain `excluded.col` assignment, so that a caller who omits a field (or
  passes `None`) on a duplicate-matching insert leaves the existing row's
  value untouched, instead of unconditionally blanking it out. A caller who
  wants to actually clear a field to `None` cannot do so via this upsert
  path; that is an accepted trade-off, since the common case (re-running an
  ingestion script with the same or partial metadata) should not silently
  destroy previously-recorded values.

  Args:
    table: The name of the table being upserted into (used to build the
      `COALESCE(..., table.col)` fallback reference).
    *args: The columns to update when a conflicting row exists.
  """

  if not is_valid_sql_identifier(table):
    raise ValueError(f'`{table}` is not a valid SQL identifier.')
  for key in args:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  if not args:
    return 'DO NOTHING'
  else:
    update_clauses_str = ', '.join(
        f'{key} = COALESCE(excluded.{key}, {table}.{key})' for key in args
    )
    return f'DO UPDATE SET {update_clauses_str}'


def format_sql_where_conditions(
    filter_dict: config_dict.ConfigDict | None = None,
    table_prefix: str | None = None,
) -> tuple[str, list[Any]]:
  r"""Build conditions string and values list for SQL WHERE from filters.

  Args:
    filter_dict: A ConfigDict of constraints to build SQL conditions from.
    table_prefix: An optional table prefix to prepend to each column name.

  Returns:
    A tuple of: a formatted string of AND-joined conditions, and a list of
    corresponding values. Safe to use in SQL WHERE statements.
  """

  if table_prefix and not is_valid_sql_identifier(table_prefix):
    raise ValueError(
        f'Table prefix `{table_prefix}` is not a valid SQL identifier.'
    )

  if not filter_dict:
    return '', []

  supported_ops = {
      'eq',
      'neq',
      'lt',
      'lte',
      'gt',
      'gte',
      'isin',
      'notin',
      'range',
      'approx',
  }

  conditions = []
  values = []

  for op_name, op_filters in filter_dict.items():

    if op_name not in supported_ops:
      raise ValueError(
          f'Unsupported operation: `{op_name}`. Supported filtering'
          f' operations are: {supported_ops}.'
      )
    if not isinstance(op_filters, config_dict.ConfigDict):
      raise ValueError(f'`{op_name}` value must be a ConfigDict.')

    for key, value in op_filters.items():
      if key == 'offsets':
        # `offsets` is not a real column in this schema (it is split across
        # `start_time`/`end_time`), but the base HopliteDBInterface's
        # duplicate-handling helpers (`_handle_window_duplicates`,
        # `_handle_annotation_duplicates`) always filter on a literal
        # `offsets` key with `approx` semantics, so it must be handled here.
        if op_name != 'approx':
          raise ValueError(
              f'Unsupported operation `{op_name}` on `offsets`; only'
              ' `approx` is supported (compares against start_time and'
              ' end_time).'
          )
        start_col = f'{table_prefix}.start_time' if table_prefix else 'start_time'
        end_col = f'{table_prefix}.end_time' if table_prefix else 'end_time'
        conditions.append(f'ABS({start_col} - ?) < 1e-6')
        conditions.append(f'ABS({end_col} - ?) < 1e-6')
        values.extend(normalize_sql_value(list(value)))
        continue

      if table_prefix:
        column = f'{table_prefix}.{key}'
      else:
        column = key

      if not is_valid_sql_identifier(key):
        raise ValueError(
            f'Table column `{column}` is not a valid SQL identifier. Fix the'
            f' filter rule under the `{op_name}` operation.'
        )

      value = normalize_sql_value(value)

      if op_name == 'eq':
        if value is None:
          conditions.append(f'{column} IS NULL')
        else:
          conditions.append(f'{column} = ?')
          values.append(value)
      elif op_name == 'neq':
        if value is None:
          conditions.append(f'{column} IS NOT NULL')
        else:
          conditions.append(f'{column} != ?')
          values.append(value)
      elif op_name == 'lt':
        conditions.append(f'{column} < ?')
        values.append(value)
      elif op_name == 'lte':
        conditions.append(f'{column} <= ?')
        values.append(value)
      elif op_name == 'gt':
        conditions.append(f'{column} > ?')
        values.append(value)
      elif op_name == 'gte':
        conditions.append(f'{column} >= ?')
        values.append(value)
      elif op_name == 'isin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        placeholders = ['?'] * len(value)
        placeholders_str = ', '.join(placeholders)
        conditions.append(f'{column} IN ({placeholders_str})')
        values.extend(value)
      elif op_name == 'notin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        placeholders = ['?'] * len(value)
        placeholders_str = ', '.join(placeholders)
        conditions.append(f'{column} NOT IN ({placeholders_str})')
        values.extend(value)
      elif op_name == 'range':
        if not isinstance(value, list) or len(value) != 2:
          raise ValueError(f'`{op_name}` value must be a list of 2 elements.')
        conditions.append(f'{column} BETWEEN ? AND ?')
        values.extend(value)
      elif op_name == 'approx':
        conditions.append(f'ABS({column} - ?) < 1e-6')
        values.append(value)

  return ' AND '.join(conditions), values


@dataclasses.dataclass
class ScoreBatch:
  """A struct-of-arrays batch of scored windows, as returned by `get_scores()`.

  Mirrors the parallel-array shape `insert_deployment_scores()`/
  `get_scores_batch()` already use, rather than a DataFrame or a list of
  per-window objects: every field except `scores` and `class_labels` is a
  length-`N` array with one entry per window (`scores[i]`/`recording_ids[i]`/
  etc. all describe the same window `i`); `scores` is `[N, n_classes]` (the
  full class-score vector for each of the `N` windows); `class_labels` is
  the fixed, length-`n_classes` column vocabulary shared by every row (not
  itself indexed by window).
  """

  deployment_ids: np.ndarray
  recording_ids: np.ndarray
  start_times: np.ndarray
  end_times: np.ndarray
  scores: np.ndarray
  class_labels: Sequence[str]

  def __len__(self) -> int:
    return len(self.recording_ids)

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Get a single window's data as a dict-of-values/arrays."""
    if not 0 <= idx < len(self):
      raise IndexError(f'Index {idx} out of bounds for ScoreBatch of length {len(self)}.')
    return {
        'deployment_id': self.deployment_ids[idx],
        'recording_id': self.recording_ids[idx],
        'start_time': self.start_times[idx],
        'end_time': self.end_times[idx],
        'scores': self.scores[idx],
        'class_labels': self.class_labels,
    }

  def to_pandas(self) -> 'pd.DataFrame':
    """Convert this batch to a pandas DataFrame with one row per window.

    Each class score is a separate column, named after the class label
    """
    import pandas as pd  # pyrefly: ignore[import-error]
    metadata_df = pd.DataFrame({
        'deployment_id': self.deployment_ids,
        'recording_id': self.recording_ids,
        'start_time': self.start_times,
        'end_time': self.end_times,
    })
    score_df = pd.DataFrame(self.scores, columns=self.class_labels)
    return pd.concat([metadata_df, score_df], axis=1)


def add_deployment_metadata(
    db: 'ParquetScoresDB',
    df: 'pd.DataFrame',
    foreign_key: str = 'deployment_id',
) -> 'pd.DataFrame':
  """Left-join `deployments` columns (including any extras) onto `df`.

  A standalone helper, not tied to `ScoreBatch.to_pandas()`'s output
  specifically -- works on any DataFrame that has a `foreign_key` column of
  deployment ids (e.g. `select_scores()`'s or `sampling_effort()`'s output
  too, or a hand-built frame). Only fetches metadata for the deployment ids
  actually present in `df`, not the whole table.

  Args:
    db: The database to fetch deployment metadata from.
    df: A DataFrame with a `foreign_key` column of deployment ids.
    foreign_key: The column in `df` holding deployment ids. Must not
      collide with `'id'` (deployments' own primary-key column name, which
      the join brings in from the other side) unless `foreign_key` is
      itself `'id'`.

  Returns:
    A new DataFrame with every column of `deployments` *except* `id`
    (redundant with `foreign_key`, which already identifies the deployment,
    unless `foreign_key` is itself `'id'`) -- `name` (renamed to
    `deployment_name`, since a bare `name` is ambiguous once merged with
    other tables' columns), `project`, `latitude`, `longitude`, and any
    extra columns added via `add_extra_table_column('deployments', ...)` --
    placed first, followed by `df`'s original columns in their original
    order, left-joined on `foreign_key == deployments.id`. Rows in `df`
    whose `foreign_key` doesn't match any deployment (should not normally
    happen, since deployment ids are foreign-key-enforced) get `NaN` for
    the deployment columns rather than being dropped. If `df` already has a
    `deployment_name` column (e.g. it came from `select_scores()`/
    `select_detections()`, which always include one), the incoming one is
    dropped rather than duplicated or `_x`/`_y`-suffixed. Any other
    deployment extra column that happens to share a name with an existing
    `df` column still gets pandas' default `_x`/`_y` suffixing (see
    `pandas.DataFrame.merge`).
  """
  if foreign_key not in df.columns:
    raise ValueError(
        f'`{foreign_key}` is not a column of `df`. Available columns:'
        f' {list(df.columns)}.'
    )
  deployment_ids = [int(v) for v in df[foreign_key].unique()]
  # Query DuckDB directly (rather than going through `get_all_deployments()`
  # + `Deployment.to_kwargs()`) so the result always has every `deployments`
  # column -- `name`, `project`, `latitude`, `longitude`, extras -- even
  # when zero rows match (e.g. no `foreign_key` values in `df`, or ids that
  # don't exist): DuckDB's own schema is authoritative, unlike a
  # `pd.DataFrame` built from an empty list of dicts, which would have no
  # columns at all and make the merge below silently add nothing instead of
  # the all-`NaN` deployment columns callers expect.
  placeholders = ', '.join(['?'] * len(deployment_ids)) or 'NULL'
  deployments_df = db.db.execute(
      f'SELECT * FROM deployments WHERE id IN ({placeholders})',
      deployment_ids,
  ).fetchdf()
  # `deployments.id` is redundant with `foreign_key` once joined (both
  # identify the same deployment); renaming it to `foreign_key` before the
  # merge lets `on=foreign_key` do the join without leaving a duplicate
  # `id`/`{foreign_key}` pair or `_x`/`_y`-suffixing either of them.
  if foreign_key != 'id':
    deployments_df = deployments_df.rename(columns={'id': foreign_key})
  # `name` field is ambiguous, call it `deployment_name` for clarity -- but
  # if `df` already has a `deployment_name` column (e.g. it came from
  # `select_scores()`/`select_detections()`, which always include one),
  # drop the incoming one rather than let it collide and get `_x`/`_y`
  # suffixed, since both name the same thing.
  deployments_df = deployments_df.rename(columns={'name': 'deployment_name'})
  if 'deployment_name' in df.columns and 'deployment_name' in deployments_df.columns:
    deployments_df = deployments_df.drop(columns=['deployment_name'])
  merged = df.merge(deployments_df, on=foreign_key, how='left')
  # `merge()` appends new columns at the end; reorder so the newly-added
  # deployment columns lead, followed by `df`'s original columns, which
  # makes the merged-in metadata easy to spot at a glance rather than
  # scrolled off to the right of however many score/class columns `df` had.
  new_columns = [c for c in deployments_df.columns if c != foreign_key]
  return merged[new_columns + list(df.columns)]


def add_recording_metadata(
    db: 'ParquetScoresDB',
    df: 'pd.DataFrame',
    foreign_key: str = 'recording_id',
) -> 'pd.DataFrame':
  """Left-join `recordings` columns (including any extras) onto `df`.

  A standalone helper, not tied to `ScoreBatch.to_pandas()`'s output
  specifically -- works on any DataFrame that has a `foreign_key` column of
  recording ids. Only fetches metadata for the recording ids actually
  present in `df`, not the whole table.

  Args:
    db: The database to fetch recording metadata from.
    df: A DataFrame with a `foreign_key` column of recording ids.
    foreign_key: The column in `df` holding recording ids. Must not
      collide with `'id'` (recordings' own primary-key column name, which
      the join brings in from the other side) unless `foreign_key` is
      itself `'id'`.

  Returns:
    A new DataFrame with every column of `recordings` *except* `id`
    (redundant with `foreign_key`, unless `foreign_key` is itself `'id'`)
    -- `filename`, `datetime`, `deployment_id`, `score_chunk_path`, and any
    extra columns added via `add_extra_table_column('recordings', ...)` --
    placed first, followed by `df`'s original columns in their original
    order, left-joined on `foreign_key == recordings.id`. Rows in `df`
    whose `foreign_key` doesn't match any recording get `NaN` for the
    recording columns rather than being dropped. If `df` already has a
    `deployment_id` column (e.g.
    from a prior `add_deployment_metadata()` call, or because `df` came
    from `ScoreBatch.to_pandas()`), `recordings.deployment_id` is dropped
    rather than duplicated or `_x`/`_y`-suffixed, since it names the same
    deployment either way. Any other recording extra column that happens to
    share a name with an existing `df` column still gets pandas' default
    `_x`/`_y` suffixing (see `pandas.DataFrame.merge`).
  """
  if foreign_key not in df.columns:
    raise ValueError(
        f'`{foreign_key}` is not a column of `df`. Available columns:'
        f' {list(df.columns)}.'
    )
  recording_ids = [int(v) for v in df[foreign_key].unique()]
  # See `add_deployment_metadata()` for why this queries DuckDB directly
  # rather than going through `get_all_recordings()` + `to_kwargs()`.
  placeholders = ', '.join(['?'] * len(recording_ids)) or 'NULL'
  recordings_df = db.db.execute(
      f'SELECT * FROM recordings WHERE id IN ({placeholders})',
      recording_ids,
  ).fetchdf()
  # See `add_deployment_metadata()` for why `id` is renamed rather than
  # left to collide with `foreign_key`.
  if foreign_key != 'id':
    recordings_df = recordings_df.rename(columns={'id': foreign_key})
  if 'deployment_id' in df.columns and 'deployment_id' in recordings_df.columns:
    recordings_df = recordings_df.drop(columns=['deployment_id'])
  merged = df.merge(recordings_df, on=foreign_key, how='left')
  # See `add_deployment_metadata()` for why the newly-added columns are
  # reordered to lead rather than trail.
  new_columns = [c for c in recordings_df.columns if c != foreign_key]
  return merged[new_columns + list(df.columns)]


@dataclasses.dataclass
class ParquetScoresDB(interface.HopliteDBInterface):
  """DuckDB + per-deployment-Parquet database for wide class-score outputs.

  See the module docstring for the overall design. In short: relational
  metadata (deployments/recordings/annotations/detections/hoplite_metadata)
  lives in native DuckDB tables in a single `.duckdb` file; the wide
  per-window class-score matrices live in one Parquet file per deployment
  under `db_path/scores/`, read via DuckDB's Parquet scanning rather than
  through any table in the `.duckdb` file itself.

  Attributes:
    db_path: The path to the database directory.
    db: The DuckDB connection.
    _readonly: Whether this connection was opened read-only.
  """

  db_path: epath.Path
  db: duckdb.DuckDBPyConnection
  _readonly: bool = False

  @property
  def duckdb_path(self) -> epath.Path:
    return self.db_path / HOPLITE_FILENAME

  @property
  def scores_dir(self) -> epath.Path:
    return self.db_path / SCORES_DIRNAME

  def deployment_parquet_path(self, deployment_id: int) -> epath.Path:
    """Path to a deployment's first (or only) score-chunk Parquet file.

    Kept for backwards compatibility / discoverability; a deployment written
    across multiple `insert_deployment_scores()` calls has additional chunk
    files alongside this one (`deployment_{id}_1.parquet`,
    `deployment_{id}_2.parquet`, ...) -- see `_deployment_chunk_paths()` for
    the full set.
    """
    return self.scores_dir / f'deployment_{deployment_id}_0.parquet'

  def _deployment_chunk_paths(self, deployment_id: int) -> list[epath.Path]:
    """All existing score-chunk Parquet file paths for a deployment.

    Derived from `recordings.score_chunk_path` (the source of truth for
    where a recording's scores live -- see PARQUET_DUCKDB_PLAN.md, Q9)
    rather than a directory listing, so this reflects exactly the chunks
    that are actually referenced by a recording, in a stable order.
    """
    rows = self.db.execute(
        'SELECT DISTINCT score_chunk_path FROM recordings'
        ' WHERE deployment_id = ? AND score_chunk_path IS NOT NULL'
        ' ORDER BY score_chunk_path',
        [int(deployment_id)],
    ).fetchall()
    return [epath.Path(row[0]) for row in rows]

  def _next_chunk_path(self, deployment_id: int) -> epath.Path:
    """The path a new score chunk for `deployment_id` should be written to.

    Numbered by globbing `scores_dir` for the highest existing numeric
    suffix and adding 1, rather than by counting rows in `recordings` --
    this stays correct even if a prior write's Parquet file landed on disk
    (atomic rename succeeded) but the follow-up `recordings.score_chunk_path`
    update never completed (e.g. a crash in between; see
    `insert_deployment_scores()`), which would otherwise make a stale,
    unreferenced chunk file collide with the next chunk number chosen.
    """
    deployment_id = int(deployment_id)
    prefix = f'deployment_{deployment_id}_'
    existing_suffixes = []
    if self.scores_dir.exists():  # pyrefly: ignore[unsupported-operation]
      for path in self.scores_dir.glob(f'{prefix}*.parquet'):  # pyrefly: ignore[unsupported-operation]
        suffix = path.stem[len(prefix) :]
        if suffix.isdigit():
          existing_suffixes.append(int(suffix))
    next_n = max(existing_suffixes, default=-1) + 1
    return self.scores_dir / f'{prefix}{next_n}.parquet'

  @staticmethod
  def _setup_tables(db: duckdb.DuckDBPyConnection) -> None:
    """Create the DuckDB tables (idempotent).

    Args:
      db: The DuckDB connection to use.
    """

    existing = {
        row[0]
        for row in db.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'main'"
        ).fetchall()
    }
    if 'recordings' in existing:
      return

    db.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        )
        """)

    db.execute('CREATE SEQUENCE IF NOT EXISTS deployments_id_seq START 1')
    db.execute("""
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY DEFAULT nextval('deployments_id_seq'),
            name VARCHAR NOT NULL,
            project VARCHAR NOT NULL,
            latitude DOUBLE,
            longitude DOUBLE,
            UNIQUE (name, project)
        )
        """)

    db.execute('CREATE SEQUENCE IF NOT EXISTS recordings_id_seq START 1')
    db.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY DEFAULT nextval('recordings_id_seq'),
            filename VARCHAR NOT NULL,
            datetime TIMESTAMPTZ,
            deployment_id INTEGER REFERENCES deployments(id),
            score_chunk_path VARCHAR,
            UNIQUE (filename, deployment_id)
        )
        """)

    db.execute('CREATE SEQUENCE IF NOT EXISTS annotations_id_seq START 1')
    db.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY DEFAULT nextval('annotations_id_seq'),
            recording_id INTEGER NOT NULL REFERENCES recordings(id),
            start_time DOUBLE NOT NULL,
            end_time DOUBLE NOT NULL,
            label VARCHAR NOT NULL,
            label_type INTEGER NOT NULL,
            provenance VARCHAR NOT NULL
        )
        """)

    db.execute('CREATE SEQUENCE IF NOT EXISTS detections_id_seq START 1')
    db.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY DEFAULT nextval('detections_id_seq'),
            recording_id INTEGER NOT NULL REFERENCES recordings(id),
            start_time DOUBLE NOT NULL,
            end_time DOUBLE NOT NULL,
            class_name VARCHAR NOT NULL,
            score DOUBLE NOT NULL,
            UNIQUE (recording_id, start_time, end_time, class_name)
        )
        """)

    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_recordings_deployment_id
        ON recordings(deployment_id)
        """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations_recording_id
        ON annotations(recording_id)
        """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_recording_id
        ON detections(recording_id)
        """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_class_name
        ON detections(class_name)
        """)

  @staticmethod
  def _get_all_metadata(
      db: duckdb.DuckDBPyConnection,
  ) -> config_dict.ConfigDict:
    """Get all key-value pairs from the metadata table."""
    rows = db.execute('SELECT key, value FROM hoplite_metadata').fetchall()
    return config_dict.ConfigDict({k: json.loads(v) for k, v in rows})

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      db_path: str,
      class_labels: Sequence[str] | None = None,
      readonly: bool = False,
  ) -> 'ParquetScoresDB':
    """Connect to and, if needed, initialize the database.

    Args:
      db_path: The path to the database directory.
      class_labels: The (fixed, ordered) class vocabulary for scores stored
        in this database. Required the first time a database is created; if
        the database already exists, this is validated against (if
        provided) or loaded from `hoplite_metadata`.
      readonly: If True, opens the database in read-only mode.

    Raises:
      ValueError: If `class_labels` is inconsistent with the DB or if no
        class vocabulary is found in the DB and none was provided.

    Returns:
      A new instance of the database.
    """

    db_path = epath.Path(db_path)  # pyrefly: ignore[bad-assignment]
    if not readonly:
      db_path.mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[missing-attribute]
      (db_path / SCORES_DIRNAME).mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[unsupported-operation]
    duckdb_path = db_path / HOPLITE_FILENAME  # pyrefly: ignore[unsupported-operation]

    db = duckdb.connect(duckdb_path.as_posix(), read_only=readonly)
    # Pin the session timezone to UTC so that datetime handling is
    # deterministic and portable across machines/CI, rather than depending
    # on the host's local timezone (DuckDB's default). Naive (no tzinfo)
    # Python `datetime` values are rejected outright at the API boundary
    # (see `require_timezone_aware()`) rather than relying on this session
    # setting to interpret them; this pin instead governs how `TIMESTAMPTZ`
    # values are normalized/displayed for the tz-aware datetimes callers do
    # pass in.
    db.execute("SET TimeZone='UTC'")

    if not readonly:
      cls._setup_tables(db)

    metadata = cls._get_all_metadata(db)
    existing_class_labels = (
        list(metadata[CLASS_LABELS_KEY].labels)
        if CLASS_LABELS_KEY in metadata
        else None
    )
    if (
        existing_class_labels is not None
        and class_labels is not None
        and existing_class_labels != list(class_labels)
    ):
      raise ValueError(
          '`class_labels` was provided, but a different class vocabulary'
          ' already exists in the DB. The class vocabulary is immutable for'
          ' the lifetime of a ParquetScoresDB.'
      )
    if existing_class_labels is not None:
      class_labels = existing_class_labels
    elif class_labels is None:
      raise ValueError(
          'No class vocabulary was found in DB and none was provided.'
      )

    hoplite_db = cls(
        db_path=db_path,  # pyrefly: ignore[bad-argument-type]
        db=db,
        _readonly=readonly,
    )

    if not readonly and CLASS_LABELS_KEY not in metadata:
      hoplite_db.insert_metadata(
          CLASS_LABELS_KEY, config_dict.ConfigDict({'labels': list(class_labels)})
      )
      hoplite_db.commit()

    return hoplite_db

  @property
  def classes(self) -> Sequence[str]:
    """Get the fixed class vocabulary for this database."""
    return tuple(self.get_metadata(CLASS_LABELS_KEY).labels)

  # ---------------------------------------------------------------------
  # Extra-column handling.
  # ---------------------------------------------------------------------

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    """Add an extra column to a table in the database."""

    if table_name not in _METADATA_TABLES:
      raise ValueError(f'Table `{table_name}` does not exist.')
    if not is_valid_sql_identifier(column_name):
      raise ValueError(f'Column `{column_name}` is not a valid SQL identifier.')
    if not isinstance(column_type, type):
      raise ValueError(f'Column type `{column_type}` must be a type.')
    if column_type not in PYTHON_TYPE_TO_SQL_TYPE:
      raise ValueError(
          f'Column type `{column_type.__name__}` is not supported. Use one'
          f' of: {", ".join([key.__name__ for key in PYTHON_TYPE_TO_SQL_TYPE.keys()])}'
      )

    existing_columns = {
        row[1]
        for row in self.db.execute(f'PRAGMA table_info({table_name})').fetchall()
    }
    if column_name in existing_columns:
      return

    self.db.execute(f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name} {PYTHON_TYPE_TO_SQL_TYPE[column_type]}
        """)

    self.__dict__.pop('_extra_table_columns', None)

  @functools.cached_property
  def _extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""
    extra_columns = {t: {} for t in _METADATA_TABLES}
    for table in _METADATA_TABLES:
      columns_info = self.db.execute(f'PRAGMA table_info({table})').fetchall()
      for col_info in columns_info:
        # col_info: cid, name, type, notnull, dflt_value, pk
        col_name = col_info[1]
        col_type = col_info[2]
        if col_name not in _DEFAULT_TABLE_COLUMNS[table]:
          try:
            extra_columns[table][col_name] = SQL_TYPE_TO_PYTHON_TYPE[col_type]
          except KeyError as e:
            raise ValueError(
                f'Unsupported column type {col_type} for column '
                f'{col_name} in table {table}'
            ) from e
    return extra_columns

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""
    return self._extra_table_columns

  def _validate_filter_columns(
      self,
      table: str,
      *filter_dicts: config_dict.ConfigDict | None,
  ) -> None:
    """Raise a clear `ValueError` if a filter references an unknown column.

    Without this, an unknown column name in a filter (e.g. `deployment_id`
    instead of `id` when filtering `deployments`, since a row's own primary
    key is always called `id` -- `deployment_id` only exists as a foreign
    key on `recordings`/`detections`) would surface as a raw
    `duckdb.BinderException` deep inside the generated SQL, which does not
    say which of the filter's columns was the problem or what the valid
    columns actually are.

    Args:
      table: The table the filter(s) will be applied to.
      *filter_dicts: The filter `ConfigDict`s to validate (any may be
        `None`, in which case it is skipped).
    """
    valid_columns = (
        _DEFAULT_TABLE_COLUMNS[table]
        | self._extra_table_columns[table].keys()
        | _FILTER_PSEUDO_COLUMNS.get(table, set())
    )
    for filter_dict in filter_dicts:
      if not filter_dict:
        continue
      for _, op_filters in filter_dict.items():
        for column in op_filters.keys():
          if column not in valid_columns:
            raise ValueError(
                f'Unknown column `{column}` in filter for table'
                f' `{table}`. Valid columns are:'
                f' {sorted(valid_columns)}.'
            )

  def _validate_columns(self, table: str, columns: Sequence[str]) -> None:
    """Raise a clear `ValueError` if `columns` references an unknown column.

    Like `_validate_filter_columns()`, but for a plain list of column names
    rather than a filter dict -- used by `by_deployment_fields` in
    `select_scores()`/`select_detections()`.

    Args:
      table: The table `columns` are expected to be columns of.
      columns: The column names to validate.
    """
    valid_columns = (
        _DEFAULT_TABLE_COLUMNS[table] | self._extra_table_columns[table].keys()
    )
    for column in columns:
      if column not in valid_columns:
        raise ValueError(
            f'Unknown column `{column}` for table `{table}`. Valid columns'
            f' are: {sorted(valid_columns)}.'
        )

  # ---------------------------------------------------------------------
  # Transactions / connection management.
  # ---------------------------------------------------------------------

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    self.db.commit()

  def rollback(self) -> None:
    """Rollback any pending transactions to the database."""
    self.db.rollback()

  def thread_split(self) -> 'ParquetScoresDB':
    """Get a new instance of the database for use in another thread.

    DuckDB's Python binding refuses to open any second `duckdb.connect()`
    connection (read-only or not) to the same database file while a first
    connection from the same process is already open -- unlike SQLite,
    where a second `sqlite3.connect()` to the same file from another thread
    works via WAL mode. The supported way to get an independent,
    thread-safe connection to the same in-process database is
    `DuckDBPyConnection.cursor()`, which is what this uses.

    The returned instance shares the same underlying data/transactions as
    `self` (including read-only-ness), but NOT session-level settings --
    `cursor()` starts a fresh session state, so e.g. the `TimeZone` pin set
    in `create()` must be re-applied explicitly here, or datetime handling
    on the split copy would silently depend on the host's local timezone
    instead. Per Q1/Q7 in PARQUET_DUCKDB_PLAN.md, only one process should be
    treated as the writer for a given database file regardless.

    Returns:
      A new `ParquetScoresDB` instance backed by an independent cursor on
      the same underlying DuckDB connection.
    """
    split_db = self.db.cursor()
    split_db.execute("SET TimeZone='UTC'")
    return dataclasses.replace(self, db=split_db)

  # ---------------------------------------------------------------------
  # hoplite_metadata.
  # ---------------------------------------------------------------------

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""
    json_coded = value.to_json()
    self.db.execute(
        """
        INSERT INTO hoplite_metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT (key)
        DO UPDATE SET value = excluded.value
        """,
        [key, json_coded],
    )

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table."""
    if key is None:
      return self._get_all_metadata(self.db)

    result = self.db.execute(
        'SELECT value FROM hoplite_metadata WHERE key = ?', [key]
    ).fetchone()
    if result is None:
      raise KeyError(f'Metadata key not found: {key}')
    return config_dict.ConfigDict(json.loads(result[0]))

  def remove_metadata(self, key: str | None) -> None:
    """Remove a key-value pair from the metadata table."""
    if key is None:
      self.db.execute('DELETE FROM hoplite_metadata')
      return

    result = self.db.execute(
        'DELETE FROM hoplite_metadata WHERE key = ? RETURNING key', [key]
    ).fetchall()
    if not result:
      raise KeyError(f'Metadata key not found: {key}')

  def _upsert_row(
      self,
      table: str,
      conflict_columns: Sequence[str],
      update_columns: Sequence[str],
      insert_kwargs: dict[str, Any],
      error_context: str,
  ) -> int:
    """Insert-or-update a row, working around a DuckDB upsert limitation.

    Tries a normal `INSERT ... ON CONFLICT (conflict_columns) DO UPDATE`
    first (a single round-trip, and the common case since most inserts are
    genuinely new rows). DuckDB refuses that upsert form specifically when
    the conflicting row is already referenced by a foreign key in another
    table (e.g. a `deployments` row that already has `recordings`), even
    when the update is a no-op -- see `_is_fk_referenced_upsert_error()`. In
    that case, falls back to an explicit lookup-by-conflict-key followed by
    a plain `UPDATE ... WHERE id = ?`, which DuckDB does allow against
    FK-referenced rows.

    Args:
      table: The table to upsert into.
      conflict_columns: The columns forming the table's uniqueness
        constraint (used for `ON CONFLICT` and, in the fallback path, the
        lookup `WHERE` clause).
      update_columns: The columns to update when a conflicting row exists.
      insert_kwargs: The full set of column values for the insert.
      error_context: Extra text appended to the `RuntimeError` raised if the
        statement fails outright (e.g. a bad foreign key elsewhere in
        `insert_kwargs`).

    Returns:
      The id of the inserted or updated row.
    """
    columns_str, placeholders_str, values = format_sql_insert_values(
        **insert_kwargs
    )
    update_clause_str = format_sql_update_on_conflict(table, *update_columns)
    conflict_columns_str = ', '.join(conflict_columns)

    try:
      result = self.db.execute(
          f"""
          INSERT INTO {table} {columns_str}
          VALUES {placeholders_str}
          ON CONFLICT ({conflict_columns_str})
          {update_clause_str}
          RETURNING id
          """,
          values,
      ).fetchone()
    except duckdb.Error as e:
      if not _is_fk_referenced_upsert_error(e):
        raise RuntimeError(
            f'Error inserting into `{table}`. {error_context}'
        ) from e

      # Fall back to an explicit lookup + plain UPDATE, which DuckDB allows
      # against a row that is referenced by a foreign key elsewhere.
      lookup_conditions = ' AND '.join(f'{c} = ?' for c in conflict_columns)
      lookup_values = [insert_kwargs[c] for c in conflict_columns]
      existing = self.db.execute(
          f'SELECT id FROM {table} WHERE {lookup_conditions}', lookup_values
      ).fetchone()
      if existing is None:
        # The conflict was against a different unique constraint than the
        # one we assumed; re-raise the original error.
        raise RuntimeError(
            f'Error inserting into `{table}`. {error_context}'
        ) from e

      row_id = existing[0]
      if update_columns:
        # Same COALESCE-with-existing-value semantics as the fast-path
        # upsert (see `format_sql_update_on_conflict`): a caller who omits
        # a field (or passes `None`) leaves the existing value untouched.
        set_clause = ', '.join(
            f'{c} = COALESCE(?, {table}.{c})' for c in update_columns
        )
        update_values = [insert_kwargs[c] for c in update_columns] + [row_id]
        self.db.execute(
            f'UPDATE {table} SET {set_clause} WHERE id = ?', update_values
        )
      return row_id

    if result is None:
      raise RuntimeError(f'Error inserting into `{table}`. {error_context}')
    return result[0]

  # ---------------------------------------------------------------------
  # deployments.
  # ---------------------------------------------------------------------

  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a deployment into the database."""
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['deployments']:
        self.add_extra_table_column('deployments', key, type(value))

    insert_kwargs = dict(
        name=name,
        project=project,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )
    return self._upsert_row(
        table='deployments',
        conflict_columns=['name', 'project'],
        update_columns=[
            'latitude',
            'longitude',
            *self._extra_table_columns['deployments'].keys(),
        ],
        insert_kwargs=insert_kwargs,
        error_context='',
    )

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
    """Get a deployment from the database."""
    deployment_id = int(deployment_id)

    cursor = self.db.execute(
        'SELECT * FROM deployments WHERE id = ?', [deployment_id]
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Deployment id not found: {deployment_id}')

    columns = [col[0] for col in cursor.description]
    return datatypes.Deployment(**dict(zip(columns, result)))

  def remove_deployment(self, deployment_id: int) -> None:
    """Remove a deployment from the database.

    This also removes all recordings/annotations/detections belonging to the
    deployment (DuckDB does not support `ON DELETE CASCADE`, so this is done
    explicitly), and deletes all of the deployment's score-chunk Parquet
    files outright (rather than leaving orphaned rows behind -- see
    PARQUET_DUCKDB_PLAN.md, Q6/Q9).
    """
    deployment_id = int(deployment_id)

    # Looked up before `recordings` rows are deleted below, since
    # `_deployment_chunk_paths()` reads `recordings.score_chunk_path`.
    chunk_paths = self._deployment_chunk_paths(deployment_id)

    recording_ids = [
        row[0]
        for row in self.db.execute(
            'SELECT id FROM recordings WHERE deployment_id = ?',
            [deployment_id],
        ).fetchall()
    ]
    if recording_ids:
      placeholders = ', '.join(['?'] * len(recording_ids))
      self.db.execute(
          f'DELETE FROM annotations WHERE recording_id IN ({placeholders})',
          recording_ids,
      )
      self.db.execute(
          f'DELETE FROM detections WHERE recording_id IN ({placeholders})',
          recording_ids,
      )
      self.db.execute(
          f'DELETE FROM recordings WHERE id IN ({placeholders})',
          recording_ids,
      )

    result = self.db.execute(
        'DELETE FROM deployments WHERE id = ? RETURNING id', [deployment_id]
    ).fetchall()
    if not result:
      raise KeyError(f'Deployment id not found: {deployment_id}')

    for chunk_path in chunk_paths:
      if chunk_path.exists():  # pyrefly: ignore[unsupported-operation]
        chunk_path.unlink()  # pyrefly: ignore[unsupported-operation]

  # ---------------------------------------------------------------------
  # recordings.
  # ---------------------------------------------------------------------

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a recording into the database.

    Args:
      filename: The filename of the recording.
      datetime: The (optional) datetime of the recording. Must be
        timezone-aware (e.g. `dt.datetime(..., tzinfo=dt.timezone.utc)`) --
        a naive datetime is rejected, since storing it would otherwise
        require silently assuming a timezone.
      deployment_id: The (optional) id of the deployment this recording
        belongs to.
      **kwargs: Additional keyword arguments to pass to the recording.

    Returns:
      The id of the inserted (or, on a duplicate `(filename,
      deployment_id)` match, the existing) recording.
    """
    require_timezone_aware(datetime, 'datetime')
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['recordings']:
        self.add_extra_table_column('recordings', key, type(value))

    insert_kwargs = dict(
        filename=filename,
        datetime=datetime,
        deployment_id=deployment_id,
        **kwargs,
    )
    return self._upsert_row(
        table='recordings',
        conflict_columns=['filename', 'deployment_id'],
        update_columns=[
            'datetime',
            *self._extra_table_columns['recordings'].keys(),
        ],
        insert_kwargs=insert_kwargs,
        error_context='Check that the deployment_id exists.',
    )

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    """Get a recording from the database."""
    recording_id = int(recording_id)

    cursor = self.db.execute(
        'SELECT * FROM recordings WHERE id = ?', [recording_id]
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Recording id not found: {recording_id}')

    columns = [col[0] for col in cursor.description]
    return datatypes.Recording(**dict(zip(columns, result)))

  def remove_recording(self, recording_id: int) -> None:
    """Not implemented.

    Deleting a single recording's scores would require rewriting the whole
    of its deployment's Parquet file (Parquet has no row-level delete); see
    PARQUET_DUCKDB_PLAN.md, Q6. Use `remove_deployment()` instead.
    """
    raise NotImplementedError(
        'remove_recording() is not supported by ParquetScoresDB, since'
        ' deleting a single recording would require rewriting its whole'
        " deployment's Parquet file. Use remove_deployment() instead."
    )

  # ---------------------------------------------------------------------
  # windows -- not supported; see module docstring.
  # ---------------------------------------------------------------------

  def insert_window(
      self,
      recording_id: int,
      offsets: list[float],
      embedding: np.ndarray | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table -- scores are written in bulk'
        ' via insert_deployment_scores() and addressed by (recording_id,'
        ' start_time, end_time), not by a per-window id. See'
        ' PARQUET_DUCKDB_PLAN.md.'
    )

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table. Use get_scores_batch() or'
        ' select_scores()/get_scores_batch() to read scores, keyed by (recording_id,'
        ' start_time, end_time).'
    )

  def get_window_annotations(
      self, window_id: int, label: str | None = None
  ) -> Sequence[datatypes.Annotation]:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table. Use get_all_annotations()'
        ' with a recordings_filter/filter, or get_detection_annotations()'
        ' for a specific detection.'
    )

  def get_embedding(self, window_id: int) -> np.ndarray:
    raise NotImplementedError(
        'ParquetScoresDB stores class-score vectors, not embeddings, and'
        ' has no per-window id to look up by. Use get_scores_batch() or'
        ' select_scores() instead.'
    )

  def remove_window(self, window_id: int) -> None:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table. Use remove_deployment() to'
        ' delete an entire deployment (metadata + its Parquet file).'
    )

  # ---------------------------------------------------------------------
  # annotations.
  # ---------------------------------------------------------------------

  def insert_annotation(
      self,
      recording_id: int,
      offsets: list[float],
      label: str,
      label_type: datatypes.LabelType,
      provenance: str,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    """Insert an annotation into the database."""
    duplicate_id = self._handle_annotation_duplicates(
        recording_id, offsets, label, label_type, provenance, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    for key, value in kwargs.items():
      if key not in self._extra_table_columns['annotations']:
        self.add_extra_table_column('annotations', key, type(value))

    columns_str, placeholders_str, values = format_sql_insert_values(
        recording_id=recording_id,
        start_time=offsets[0],
        end_time=offsets[1],
        label=label,
        label_type=label_type,
        provenance=provenance,
        **kwargs,
    )
    try:
      result = self.db.execute(
          f"""
          INSERT INTO annotations {columns_str}
          VALUES {placeholders_str}
          RETURNING id
          """,
          values,
      ).fetchone()
    except duckdb.Error as e:
      raise RuntimeError(
          'Error inserting the annotation into the database. Check that the'
          ' recording_id exists.'
      ) from e

    if result is None:
      raise RuntimeError('Error inserting the annotation into the database.')
    return result[0]

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    """Get an annotation from the database."""
    annotation_id = int(annotation_id)

    cursor = self.db.execute(
        'SELECT * FROM annotations WHERE id = ?', [annotation_id]
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Annotation id not found: {annotation_id}')

    columns = [col[0] for col in cursor.description]
    row = dict(zip(columns, result))
    offsets = [row.pop('start_time'), row.pop('end_time')]
    annotation = datatypes.Annotation(offsets=offsets, **row)
    annotation.label_type = datatypes.LabelType(annotation.label_type)
    return annotation

  def remove_annotation(self, annotation_id: int) -> None:
    """Remove an annotation from the database."""
    annotation_id = int(annotation_id)

    result = self.db.execute(
        'DELETE FROM annotations WHERE id = ? RETURNING id', [annotation_id]
    ).fetchall()
    if not result:
      raise KeyError(f'Annotation id not found: {annotation_id}')

  # ---------------------------------------------------------------------
  # detections (new; see module docstring).
  # ---------------------------------------------------------------------

  def insert_detection(
      self,
      recording_id: int,
      offsets: list[float],
      class_name: str,
      score: float,
      handle_duplicates: Literal['overwrite', 'skip', 'error'] = 'overwrite',
      **kwargs: Any,
  ) -> int:
    """Insert a single detection into the database.

    A detection's identity is `(recording_id, offsets, class_name)` -- the
    `detections` table has a `UNIQUE(recording_id, start_time, end_time,
    class_name)` constraint, so `score` (and any extra columns) are treated
    as updateable values, not part of what makes two detections "the same".
    This means true duplicate-*allowing* insertion (two rows with the exact
    same identity) is not supported: the whole point of the constraint is
    to prevent that. Use `handle_duplicates` to control what happens when a
    matching detection already exists.

    Args:
      recording_id: The recording this detection belongs to.
      offsets: The `[start_time, end_time]` of the detection.
      class_name: The single class this detection is for.
      score: The score that triggered this detection.
      handle_duplicates: How to handle a detection matching another one's
        `(recording_id, offsets, class_name)`. If `'overwrite'` (the
        default), the existing detection's `score` (and any extra columns)
        are updated to the new values. If `'skip'`, the existing detection
        is left untouched and its id is returned. If `'error'`, a
        `RuntimeError` is raised.
      **kwargs: Additional keyword arguments to pass to the detection.

    Returns:
      The id of the inserted (or, on a duplicate match with `'overwrite'`
      or `'skip'`, the existing) detection.

    Raises:
      RuntimeError: If `handle_duplicates` is `'error'` and a matching
        detection already exists, or if the insert fails for another
        reason (e.g. `recording_id` does not exist).
    """
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['detections']:
        self.add_extra_table_column('detections', key, type(value))

    insert_kwargs = dict(
        recording_id=recording_id,
        start_time=offsets[0],
        end_time=offsets[1],
        class_name=class_name,
        score=score,
        **kwargs,
    )
    conflict_columns = ['recording_id', 'start_time', 'end_time', 'class_name']

    if handle_duplicates == 'overwrite':
      return self._upsert_row(
          table='detections',
          conflict_columns=conflict_columns,
          update_columns=['score', *self._extra_table_columns['detections'].keys()],
          insert_kwargs=insert_kwargs,
          error_context='Check that the recording_id exists.',
      )

    columns_str, placeholders_str, values = format_sql_insert_values(
        **insert_kwargs
    )
    conflict_columns_str = ', '.join(conflict_columns)
    on_conflict_clause = (
        f'ON CONFLICT ({conflict_columns_str}) DO NOTHING'
        if handle_duplicates == 'skip'
        else ''
    )
    try:
      result = self.db.execute(
          f"""
          INSERT INTO detections {columns_str}
          VALUES {placeholders_str}
          {on_conflict_clause}
          RETURNING id
          """,
          values,
      ).fetchone()
    except duckdb.Error as e:
      if handle_duplicates == 'error' and _is_unique_constraint_error(e):
        existing = self.db.execute(
            f"""
            SELECT id FROM detections
            WHERE {' AND '.join(f'{c} = ?' for c in conflict_columns)}
            """,
            [insert_kwargs[c] for c in conflict_columns],
        ).fetchone()
        raise RuntimeError(
            f'Duplicate detection found (id = {existing[0]}), but'
            " `handle_duplicates` is set to 'error'."
        ) from e
      raise RuntimeError(
          'Error inserting the detection into the database. Check that the'
          ' recording_id exists.'
      ) from e

    if result is not None:
      return result[0]

    # `handle_duplicates == 'skip'` and `DO NOTHING` fired (no row
    # returned): look up and return the existing detection's id.
    existing = self.db.execute(
        f"""
        SELECT id FROM detections
        WHERE {' AND '.join(f'{c} = ?' for c in conflict_columns)}
        """,
        [insert_kwargs[c] for c in conflict_columns],
    ).fetchone()
    if existing is None:
      raise RuntimeError('Error inserting the detection into the database.')
    return existing[0]

  def insert_detections_batch(
      self,
      detections_batch: Sequence[dict[str, Any]],
      handle_duplicates: Literal['overwrite', 'skip', 'error'] = 'overwrite',
  ) -> Sequence[int]:
    """Insert a batch of detections into the database.

    Args:
      detections_batch: A sequence of dicts, each with the same keys as the
        arguments of `insert_detection()` (except `handle_duplicates`,
        which applies uniformly via the parameter below unless a dict
        overrides it with its own `handle_duplicates` key).
      handle_duplicates: Same as `insert_detection()`'s `handle_duplicates`,
        applied to every detection in the batch.

    Returns:
      The ids of the inserted detections, in the same order as the input.
    """
    return [
        self.insert_detection(
            handle_duplicates=handle_duplicates, **detection_kwargs
        )
        if 'handle_duplicates' not in detection_kwargs
        else self.insert_detection(**detection_kwargs)
        for detection_kwargs in detections_batch
    ]

  def get_detection(self, detection_id: int) -> datatypes.Detection:
    """Get a detection from the database."""
    detection_id = int(detection_id)

    cursor = self.db.execute(
        'SELECT * FROM detections WHERE id = ?', [detection_id]
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Detection id not found: {detection_id}')

    columns = [col[0] for col in cursor.description]
    row = dict(zip(columns, result))
    offsets = [row.pop('start_time'), row.pop('end_time')]
    return datatypes.Detection(offsets=offsets, **row)

  def remove_detection(self, detection_id: int) -> None:
    """Remove a detection from the database."""
    detection_id = int(detection_id)

    result = self.db.execute(
        'DELETE FROM detections WHERE id = ? RETURNING id', [detection_id]
    ).fetchall()
    if not result:
      raise KeyError(f'Detection id not found: {detection_id}')

  def get_detection_annotations(
      self, detection_id: int, label: str | None = None
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations intersecting the given detection."""
    detection = self.get_detection(detection_id)
    d_start, d_end = detection.offsets

    query = """
        SELECT *
        FROM annotations
        WHERE recording_id = ?
        AND start_time < ?
        AND end_time > ?
        """
    params = [detection.recording_id, d_end, d_start]
    if label is not None:
      query += ' AND label = ?'
      params.append(label)  # pyrefly: ignore[bad-argument-type]

    cursor = self.db.execute(query, params)
    annotations = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      row = dict(zip(columns, result))
      offsets = [row.pop('start_time'), row.pop('end_time')]
      annotation = datatypes.Annotation(offsets=offsets, **row)
      annotation.label_type = datatypes.LabelType(annotation.label_type)
      annotations.append(annotation)
    return annotations

  def get_all_detections(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      recordings_filter: config_dict.ConfigDict | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Detection]:
    """Get all detections from the database, with optional filtering."""
    self._validate_filter_columns('detections', filter)
    self._validate_filter_columns('recordings', recordings_filter)
    self._validate_filter_columns('deployments', deployments_filter)
    from_clause = 'FROM detections'
    query_tables = {'detections'}
    if recordings_filter:
      query_tables |= {'recordings'}
    if deployments_filter:
      query_tables |= {'recordings', 'deployments'}
    if 'recordings' in query_tables:
      from_clause += (
          ' JOIN recordings ON detections.recording_id = recordings.id'
      )
    if 'deployments' in query_tables:
      from_clause += (
          ' JOIN deployments ON recordings.deployment_id = deployments.id'
      )

    conditions, values = tuple(
        zip(*[
            format_sql_where_conditions(filter, table_prefix='detections'),
            format_sql_where_conditions(
                recordings_filter, table_prefix='recordings'
            ),
            format_sql_where_conditions(
                deployments_filter, table_prefix='deployments'
            ),
        ])
    )
    conditions_str = ' AND '.join(c for c in conditions if c)
    values = [v for vs in values for v in vs]
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''

    select_clause = (
        'SELECT DISTINCT detections.*'
        if len(query_tables) > 1
        else 'SELECT detections.*'
    )
    cursor = self.db.execute(
        f'{select_clause} {from_clause} {where_clause}', values
    )

    detections = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      row = dict(zip(columns, result))
      offsets = [row.pop('start_time'), row.pop('end_time')]
      detections.append(datatypes.Detection(offsets=offsets, **row))
    return detections

  # ---------------------------------------------------------------------
  # Aggregate / filtered accessors.
  # ---------------------------------------------------------------------

  def count_embeddings(self) -> int:
    raise NotImplementedError(
        'ParquetScoresDB has no embeddings. Use classes or query'
        ' the score Parquet files directly for counts of scored windows.'
    )

  def count_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> int:
    """Count recordings in the database, with optional filtering."""
    self._validate_filter_columns('recordings', filter)
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    return self.db.execute(
        f'SELECT COUNT(*) FROM recordings {where_clause}', values
    ).fetchone()[0]

  def count_detections(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      recordings_filter: config_dict.ConfigDict | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
  ) -> int:
    """Count detections in the database, with optional filtering.

    Args mirror `get_all_detections()`; this is `len(get_all_detections(...))`
    computed via `COUNT(*)` in SQL rather than materializing every matching
    `Detection`.
    """
    self._validate_filter_columns('detections', filter)
    self._validate_filter_columns('recordings', recordings_filter)
    self._validate_filter_columns('deployments', deployments_filter)
    from_clause = 'FROM detections'
    query_tables = {'detections'}
    if recordings_filter:
      query_tables |= {'recordings'}
    if deployments_filter:
      query_tables |= {'recordings', 'deployments'}
    if 'recordings' in query_tables:
      from_clause += (
          ' JOIN recordings ON detections.recording_id = recordings.id'
      )
    if 'deployments' in query_tables:
      from_clause += (
          ' JOIN deployments ON recordings.deployment_id = deployments.id'
      )

    conditions, values = tuple(
        zip(*[
            format_sql_where_conditions(filter, table_prefix='detections'),
            format_sql_where_conditions(
                recordings_filter, table_prefix='recordings'
            ),
            format_sql_where_conditions(
                deployments_filter, table_prefix='deployments'
            ),
        ])
    )
    conditions_str = ' AND '.join(c for c in conditions if c)
    values = [v for vs in values for v in vs]
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''

    select_clause = (
        'SELECT COUNT(DISTINCT detections.id)'
        if len(query_tables) > 1
        else 'SELECT COUNT(*)'
    )
    return self.db.execute(
        f'{select_clause} {from_clause} {where_clause}', values
    ).fetchone()[0]

  def count_windows(self, deployment_id: int | None = None) -> int:
    """Count scored windows (rows across score chunk Parquet files).

    Reads only each chunk file's Parquet footer metadata (`num_rows`), never
    the actual score data -- so this is cheap regardless of how wide the
    class vocabulary is or how many rows a deployment has, unlike a `COUNT(*)`
    query over `read_parquet(...)`, which would still need to open and scan
    the files (DuckDB can sometimes optimize a bare `COUNT(*)` this way too,
    but reading the footer directly makes the cost explicit and avoids
    relying on query-planner behavior).

    Args:
      deployment_id: If provided, count only this deployment's windows. If
        omitted, counts windows across every deployment in the database.

    Returns:
      The total number of scored windows (rows).
    """
    import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

    if deployment_id is not None:
      chunk_paths = self._deployment_chunk_paths(int(deployment_id))
    else:
      chunk_paths = [
          epath.Path(row[0])
          for row in self.db.execute(
              'SELECT DISTINCT score_chunk_path FROM recordings'
              ' WHERE score_chunk_path IS NOT NULL'
          ).fetchall()
      ]
    return sum(
        pq.ParquetFile(chunk_path.as_posix()).metadata.num_rows
        for chunk_path in chunk_paths
    )

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table/window ids. Use'
        ' select_scores()/get_scores_batch() to query scores directly.'
    )

  def get_all_projects(self) -> Sequence[str]:
    """Get all distinct projects from the database."""
    rows = self.db.execute(
        'SELECT DISTINCT project FROM deployments ORDER BY project'
    ).fetchall()
    return [row[0] for row in rows]

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    """Get all deployments from the database."""
    self._validate_filter_columns('deployments', filter)
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor = self.db.execute(f'SELECT * FROM deployments {where_clause}', values)

    columns = [col[0] for col in cursor.description]
    return [
        datatypes.Deployment(**dict(zip(columns, result)))
        for result in cursor.fetchall()
    ]

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    """Get all recordings from the database."""
    self._validate_filter_columns('recordings', filter)
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor = self.db.execute(f'SELECT * FROM recordings {where_clause}', values)

    columns = [col[0] for col in cursor.description]
    return [
        datatypes.Recording(**dict(zip(columns, result)))
        for result in cursor.fetchall()
    ]

  def get_all_windows(
      self,
      include_embedding: bool = False,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      annotations_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Window]:
    raise NotImplementedError(
        'ParquetScoresDB has no windows table. Use'
        ' select_scores()/get_scores_batch() to query scores directly,'
        ' or get_all_detections() for thresholded single-class hits.'
    )

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations from the database."""
    self._validate_filter_columns('annotations', filter)
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor = self.db.execute(
        f'SELECT * FROM annotations {where_clause}', values
    )

    annotations = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      row = dict(zip(columns, result))
      offsets = [row.pop('start_time'), row.pop('end_time')]
      annotation = datatypes.Annotation(offsets=offsets, **row)
      annotation.label_type = datatypes.LabelType(annotation.label_type)
      annotations.append(annotation)
    return annotations

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    """Get all distinct labels from the database."""
    if label_type is None:
      where_clause = ''
      values = []
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    rows = self.db.execute(
        f'SELECT DISTINCT label FROM annotations {where_clause} ORDER BY label',
        values,
    ).fetchall()
    return [row[0] for row in rows]

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Any:
    """Count each label in the database, ignoring provenance."""
    import collections  # pylint: disable=g-import-not-at-top

    if label_type is None:
      where_clause = ''
      values = []
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''

    rows = self.db.execute(
        f"""
        SELECT label, COUNT(*)
        FROM (
            SELECT DISTINCT recording_id, start_time, end_time, label, label_type
            FROM annotations
            {where_clause}
        )
        GROUP BY label
        ORDER BY label
        """,
        values,
    ).fetchall()
    return collections.Counter({row[0]: row[1] for row in rows})

  # ---------------------------------------------------------------------
  # Score storage / retrieval (Parquet-backed).
  # ---------------------------------------------------------------------

  def get_embedding_dim(self) -> int:
    """Get the width of the per-window class-score vector."""
    return len(self.classes)

  def get_embedding_dtype(self) -> type[Any]:
    """Get the score dtype. ParquetScoresDB always stores scores as float16."""
    return np.float16

  def search(
      self,
      query_embedding: np.ndarray,
      search_list_size: int,
      approximate: bool = True,
      target_score: float | None = None,
      score_fn_name: str = 'dot',
      **kwargs: Any,
  ) -> Any:
    raise NotImplementedError(
        'ParquetScoresDB has no embedding vectors and no ANN index. Use'
        ' select_scores() to query scores for a single class'
        ' across files, or get_scores_batch() to fetch score vectors for'
        ' specific (recording_id, start_time, end_time) keys.'
    )

  def insert_deployment_scores(
      self,
      deployment_id: int,
      recording_ids: Sequence[int],
      start_times: Sequence[float],
      end_times: Sequence[float],
      scores: np.ndarray,
      handle_existing_recordings: Literal['error', 'skip', 'overwrite'] = (
          'error'
      ),
  ) -> None:
    """Write class scores for a batch of a deployment's windows.

    This is the only way to write scores in `ParquetScoresDB`: there is no
    per-row/per-window insertion, since Parquet is not designed for
    single-row appends (see PARQUET_DUCKDB_PLAN.md, Q2). May be called once
    per deployment (after a single inference pass has produced scores for
    every scored clip in the deployment), or multiple times with disjoint
    recordings (e.g. to bound peak memory on a very large deployment, or a
    memory-constrained machine) -- a deployment's scores end up spread
    across one or more Parquet "chunk" files, transparently unioned by
    every read method (`select_scores`, `get_scores_batch`,
    `add_detections_from_threshold`, ...).

    IMPORTANT -- within one call, this always writes exactly one *new* chunk
    file containing exactly the `recording_ids`/`start_times`/`end_times`/
    `scores` you pass in (minus anything dropped by
    `handle_existing_recordings='skip'`, or updated in place by
    `'overwrite'` -- see below). It never appends to or merges into a chunk
    file written by an earlier call. If you need the union of two batches of
    *new* (never-before-scored) recordings in one chunk file, pass all of
    the rows in a single call; otherwise, just call this once per batch --
    each call's rows land in their own chunk, and all chunks are queried
    together automatically. The write itself is atomic per chunk (via a temp
    file and rename), so a reader will only ever see a fully-old or
    fully-new version of any one chunk file, never a partial one.

    `handle_existing_recordings` controls what happens when the batch
    includes a recording that some earlier call already wrote scores for.
    This check (and the `'skip'`/`'overwrite'` actions it can take) is keyed
    **per recording**, never per score row: a recording's scores are always
    written together by whichever call first wrote them, so there is no
    partial state where only some of a recording's windows have been
    scored. For example, with `handle_existing_recordings='overwrite'`, if
    the incoming batch contains any rows for recording 1, *every* existing
    row for recording 1 (all of its start/end-time windows, in whichever
    chunk file they currently live) is replaced by recording 1's rows in
    this batch -- this is not a row-by-row diff/merge.

    Args:
      deployment_id: The deployment these scores belong to. All
        `recording_ids` must already exist and belong to this deployment.
      recording_ids: The recording id for each scored window.
      start_times: The start offset (seconds) for each scored window.
      end_times: The end offset (seconds) for each scored window.
      scores: A `[len(recording_ids), len(classes)]` array of
        per-class scores, in the same column order as `classes`.
      handle_existing_recordings: How to handle a recording in the batch
        that already has scores from an earlier call. If `'error'` (the
        default), raises before writing anything if *any* recording in the
        batch already has scores. If `'skip'`, recordings that already have
        scores are dropped from the batch before writing; only
        never-before-scored recordings are written to a new chunk (if the
        filtered batch ends up empty, nothing is written at all). If
        `'overwrite'`, already-scored recordings' rows are updated in place
        within their existing chunk file(s) -- their `score_chunk_path`
        does not change -- and only the batch's genuinely-new recordings (if
        any) are written to a new chunk. `'overwrite'`'s cost is bounded by
        how fragmented the affected recordings' existing chunks are, not by
        the size of this batch, since every affected old chunk file must be
        read back and rewritten.

    Raises:
      ValueError: If input lengths/shapes are inconsistent, or if any
        recording in the batch already has scores and
        `handle_existing_recordings` is `'error'`.
    """
    if not handle_existing_recordings in ('error', 'skip', 'overwrite'):
      raise ValueError(
          f'`handle_existing_recordings` must be one of'
          f" 'error', 'skip', or 'overwrite', but got"
          f" {handle_existing_recordings!r}."
      )
    deployment_id = int(deployment_id)
    class_labels = self.classes
    n = len(recording_ids)
    if not (len(start_times) == n and len(end_times) == n):
      raise ValueError(
          '`recording_ids`, `start_times`, and `end_times` must all have the'
          f' same length (got {n}, {len(start_times)}, {len(end_times)}).'
      )
    if scores.shape != (n, len(class_labels)):
      raise ValueError(
          f'`scores` must have shape ({n}, {len(class_labels)}) to match'
          f' `recording_ids` and `classes`, but got'
          f' {scores.shape}.'
      )
    scores = np.asarray(scores, dtype=np.float16)

    recording_ids_arr = np.asarray(recording_ids, dtype=np.int64)
    unique_recording_ids = sorted(set(recording_ids_arr.tolist()))
    placeholders = ', '.join(['?'] * len(unique_recording_ids))
    existing_chunk_by_recording = dict(
        self.db.execute(
            'SELECT id, score_chunk_path FROM recordings'
            f' WHERE id IN ({placeholders}) AND score_chunk_path IS NOT NULL',
            unique_recording_ids,
        ).fetchall()
    ) if unique_recording_ids else {}

    if existing_chunk_by_recording and handle_existing_recordings == 'error':
      raise ValueError(
          'The following recordings already have scores written:'
          f' {sorted(existing_chunk_by_recording)}. Pass'
          " handle_existing_recordings='skip' to skip them, or"
          " ='overwrite' to replace their scores in place."
      )

    if existing_chunk_by_recording and handle_existing_recordings == (
        'overwrite'
    ):
      overwrite_mask = np.isin(
          recording_ids_arr, list(existing_chunk_by_recording)
      )
      self._overwrite_recordings_in_place(
          existing_chunk_by_recording,
          class_labels,
          recording_ids=recording_ids_arr[overwrite_mask],
          start_times=np.asarray(start_times)[overwrite_mask],
          end_times=np.asarray(end_times)[overwrite_mask],
          scores=scores[overwrite_mask],
      )

    if existing_chunk_by_recording:
      # Both 'skip' and 'overwrite' exclude already-scored recordings from
      # the new chunk written below -- 'skip' drops them outright, and
      # 'overwrite' has already updated them in their existing chunk(s).
      keep_mask = ~np.isin(
          recording_ids_arr, list(existing_chunk_by_recording)
      )
      recording_ids = recording_ids_arr[keep_mask]
      start_times = np.asarray(start_times)[keep_mask]
      end_times = np.asarray(end_times)[keep_mask]
      scores = scores[keep_mask]

    if len(recording_ids) == 0:  # pylint: disable=g-explicit-length-test
      return

    chunk_path = self._next_chunk_path(deployment_id)
    table = _build_scores_arrow_table(
        recording_ids, start_times, end_times, scores, class_labels
    )
    # NOTE(parquet_scores): write via pyarrow.parquet directly rather than
    # `self.db.from_arrow(table).to_parquet(...)` -- DuckDB's Arrow *import*
    # does not support the half-float (float16) type ("Unsupported Internal
    # Arrow Type e"), even though DuckDB's `read_parquet` reads a float16
    # Parquet column back with no issue. Only the import path is affected.
    import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

    self.scores_dir.mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[unsupported-operation]
    tmp_path = chunk_path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp_path.as_posix())
    tmp_path.replace(chunk_path)

    # Update `recordings` only after the chunk file is durably in place --
    # a crash between the rename above and this update leaves an
    # orphaned-but-harmless chunk file, never a `recordings` row pointing at
    # a chunk file that doesn't exist (see PARQUET_DUCKDB_PLAN.md, Q9).
    new_recording_ids = [int(r) for r in recording_ids]
    placeholders = ', '.join(['?'] * len(new_recording_ids))
    self.db.execute(
        'UPDATE recordings SET score_chunk_path = ?'
        f' WHERE id IN ({placeholders})',
        [chunk_path.as_posix(), *new_recording_ids],
    )

  def _overwrite_recordings_in_place(
      self,
      existing_chunk_by_recording: dict[int, str],
      class_labels: Sequence[str],
      recording_ids: np.ndarray,
      start_times: np.ndarray,
      end_times: np.ndarray,
      scores: np.ndarray,
  ) -> None:
    """Update already-scored recordings' rows in place, chunk by chunk.

    Helper for `insert_deployment_scores(handle_existing_recordings=
    'overwrite')`. `recording_ids`/`start_times`/`end_times`/`scores` here
    are already filtered down to just the rows belonging to recordings in
    `existing_chunk_by_recording` (the caller does this filtering, since it
    already has the full incoming batch in hand).

    Cost is bounded by how fragmented the affected recordings' existing
    chunks are, not by the size of the batch passed in here: every distinct
    old chunk file referenced by `existing_chunk_by_recording` must be read
    back in full and rewritten, regardless of how many of its rows are
    actually being replaced.
    """
    import pyarrow as pa  # pylint: disable=g-import-not-at-top
    import pyarrow.compute as pc  # pylint: disable=g-import-not-at-top
    import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

    recording_ids = np.asarray(recording_ids)
    chunk_to_recording_ids: dict[str, list[int]] = {}
    for recording_id, chunk_path_str in existing_chunk_by_recording.items():
      chunk_to_recording_ids.setdefault(chunk_path_str, []).append(
          recording_id
      )

    for chunk_path_str, overwritten_ids in chunk_to_recording_ids.items():
      chunk_path = epath.Path(chunk_path_str)
      # NOTE(parquet_scores): read via pyarrow.parquet directly, not DuckDB's
      # `read_parquet` -- DuckDB's Arrow *export* widens the on-disk
      # half-float (float16) score columns to float32, which would make
      # `remaining` and `new_rows_for_chunk` (built at float16 by
      # `_build_scores_arrow_table`) schema-incompatible for
      # `concat_tables` below. `pq.read_table` preserves float16 exactly, as
      # written by `pq.write_table` -- see the matching NOTE in
      # `insert_deployment_scores`.
      full_table = pq.read_table(chunk_path.as_posix())
      remaining = full_table.filter(
          pc.invert(pc.is_in(full_table['recording_id'], pa.array(overwritten_ids)))
      )

      chunk_mask = np.isin(recording_ids, overwritten_ids)
      new_rows_for_chunk = _build_scores_arrow_table(
          recording_ids[chunk_mask],
          start_times[chunk_mask],
          end_times[chunk_mask],
          scores[chunk_mask],
          class_labels,
      )
      # NOTE: `class_labels` (and therefore column order/schema) is fixed
      # for the database's lifetime (PARQUET_DUCKDB_PLAN.md, Q5/Q8), so
      # `remaining` and `new_rows_for_chunk` always share the same schema
      # and `concat_tables` needs no schema unification.
      updated_table = pa.concat_tables([remaining, new_rows_for_chunk])

      tmp_path = chunk_path.with_suffix('.parquet.tmp')
      pq.write_table(updated_table, tmp_path.as_posix())
      tmp_path.replace(chunk_path)

  def _class_column(self, class_name: str) -> str:
    """Validate a class name and return a safe, quoted SQL column reference.

    Class names commonly contain spaces/hyphens (e.g. "American Redstart"),
    which are valid Parquet column names but not valid bare SQL identifiers.
    Since `class_name` is always checked against the database's fixed,
    caller-controlled `class_labels` vocabulary first (never arbitrary user
    input), it is safe to double-quote here; the identifier is still
    escaped (doubling any embedded `"`) per standard SQL quoted-identifier
    rules, rather than relying on membership-checking alone as the escaping
    mechanism.
    """
    class_labels = self.classes
    if class_name not in class_labels:
      raise ValueError(
          f'Unknown class `{class_name}`. Must be one of the'
          " database's fixed class_labels."
      )
    return quote_sql_identifier(class_name)

  def _scores_union_query(
      self,
      class_name: str,
      deployment_ids: Sequence[int],
  ) -> str | None:
    """Build a `UNION ALL` query over per-deployment Parquet scans.

    Each branch reads only the requested class's score column (plus key
    columns) from its deployment's Parquet file via DuckDB's column
    projection; DuckDB plans the whole `UNION ALL` as a single query rather
    than as separate round-trips, so this stays a single `execute()` call
    regardless of how many deployments are involved.

    Args:
      class_name: The class to fetch scores for. Must be one of
        `classes`.
      deployment_ids: The deployments to scan. Any deployment with no score
        chunk files yet (no scores written) is silently skipped.

    Returns:
      A SQL query string with columns `deployment_id`, `recording_id`,
      `start_time`, `end_time`, `score`, or `None` if no deployment in
      `deployment_ids` has any score chunk files.
    """
    class_column = self._class_column(class_name)
    branches = []
    for deployment_id in deployment_ids:
      chunk_paths = self._deployment_chunk_paths(deployment_id)
      if not chunk_paths:
        continue
      paths_str = ', '.join(f"'{p.as_posix()}'" for p in chunk_paths)
      branches.append(f"""
          SELECT
              {int(deployment_id)} AS deployment_id,
              recording_id,
              start_time,
              end_time,
              {class_column} AS score
          FROM read_parquet([{paths_str}])
          """)
    if not branches:
      return None
    return ' UNION ALL '.join(branches)

  def _scores_union_query_all_classes(
      self,
      deployment_ids: Sequence[int],
  ) -> str | None:
    """Build a `UNION ALL` + `UNPIVOT` query over all classes at once.

    Like `_scores_union_query()`, but instead of projecting a single class
    column, this reshapes every class column into one row per `(window,
    class)` pair via `UNPIVOT` -- so a caller wanting to scan across all
    classes (e.g. `add_detections_from_threshold()` with no `class_name`)
    does not need one query per class.

    Args:
      deployment_ids: The deployments to scan. Any deployment with no score
        chunk files yet (no scores written) is silently skipped.

    Returns:
      A SQL query string with columns `deployment_id`, `recording_id`,
      `start_time`, `end_time`, `class_name`, `score`, or `None` if no
      deployment in `deployment_ids` has any score chunk files.
    """
    class_labels = self.classes
    class_columns_str = ', '.join(
        quote_sql_identifier(c) for c in class_labels
    )
    branches = []
    for deployment_id in deployment_ids:
      chunk_paths = self._deployment_chunk_paths(deployment_id)
      if not chunk_paths:
        continue
      paths_str = ', '.join(f"'{p.as_posix()}'" for p in chunk_paths)
      branches.append(f"""
          SELECT
              {int(deployment_id)} AS deployment_id,
              recording_id,
              start_time,
              end_time,
              class_name,
              score
          FROM read_parquet([{paths_str}])
          UNPIVOT (score FOR class_name IN ({class_columns_str}))
          """)
    if not branches:
      return None
    return ' UNION ALL '.join(branches)

  def _windows_union_query(
      self,
      deployment_ids: Sequence[int],
  ) -> str | None:
    """Build a `UNION ALL` query over per-deployment Parquet scans, key columns only.

    Unlike `_scores_union_query()`/`_scores_union_query_all_classes()`, this
    projects no score/class column at all -- just the columns that identify
    a window -- so counting windows (e.g. `sampling_effort()`) never reads
    or unpivots any class-score data, regardless of how many classes are in
    the vocabulary.

    Args:
      deployment_ids: The deployments to scan. Any deployment with no score
        chunk files yet (no scores written) is silently skipped.

    Returns:
      A SQL query string with columns `deployment_id`, `recording_id`,
      `start_time`, `end_time`, or `None` if no deployment in
      `deployment_ids` has any score chunk files.
    """
    branches = []
    for deployment_id in deployment_ids:
      chunk_paths = self._deployment_chunk_paths(deployment_id)
      if not chunk_paths:
        continue
      paths_str = ', '.join(f"'{p.as_posix()}'" for p in chunk_paths)
      branches.append(f"""
          SELECT
              {int(deployment_id)} AS deployment_id,
              recording_id,
              start_time,
              end_time
          FROM read_parquet([{paths_str}])
          """)
    if not branches:
      return None
    return ' UNION ALL '.join(branches)

  def _wide_scores_union_query(
      self,
      deployment_ids: Sequence[int],
  ) -> str | None:
    """Build a `UNION ALL` query over per-deployment Parquet scans, all class columns wide.

    Unlike `_scores_union_query_all_classes()`, this does not `UNPIVOT` --
    each window stays one row, with every class as its own column -- so a
    caller wanting the full score vector per window (e.g. `get_scores()`)
    gets back something directly reshapable into a `[n_windows, n_classes]`
    array, not one row per `(window, class)` pair.

    Args:
      deployment_ids: The deployments to scan. Any deployment with no score
        chunk files yet (no scores written) is silently skipped.

    Returns:
      A SQL query string with columns `deployment_id`, `recording_id`,
      `start_time`, `end_time`, plus one column per class (see `classes`),
      or `None` if no deployment in `deployment_ids` has any score chunk
      files.
    """
    class_labels = self.classes
    class_columns_str = ', '.join(
        quote_sql_identifier(c) for c in class_labels
    )
    branches = []
    for deployment_id in deployment_ids:
      chunk_paths = self._deployment_chunk_paths(deployment_id)
      if not chunk_paths:
        continue
      paths_str = ', '.join(f"'{p.as_posix()}'" for p in chunk_paths)
      branches.append(f"""
          SELECT
              {int(deployment_id)} AS deployment_id,
              recording_id,
              start_time,
              end_time,
              {class_columns_str}
          FROM read_parquet([{paths_str}])
          """)
    if not branches:
      return None
    return ' UNION ALL '.join(branches)

  def _build_date_bin_clause(
      self,
      date_bins: Literal['day', 'week']
      | Sequence[tuple[dt.datetime, str]]
      | None,
      datetime_column: str = 'recordings.datetime',
  ) -> tuple[str, str]:
    """Build the date-bin SQL expression and any join it depends on.

    Shared by `select_scores()` and `select_detections()`.

    Args:
      date_bins: Same as `select_scores()`'s `date_bins` argument.
      datetime_column: The (already-qualified) column to bin on.

    Returns:
      A tuple `(bucket_join, date_bin_expr)`: `bucket_join` is an extra SQL
      `JOIN` clause to splice into the query (empty string if none is
      needed), and `date_bin_expr` is a SQL expression for the bin label
      (`'NULL'` if `date_bins` is `None`).
    """
    if date_bins is None:
      return '', 'NULL'
    if isinstance(date_bins, str):
      return '', f"date_trunc('{date_bins}', {datetime_column})"

    # Explicit (bin_start, bin_label) edges: assign each row to the latest
    # bin whose start is <= its recording's datetime via an as-of join
    # against a small caller-supplied bin table.
    self.db.execute('DROP TABLE IF EXISTS _select_scores_buckets')
    self.db.execute(
        'CREATE TEMP TABLE _select_scores_buckets'
        ' (bucket_start TIMESTAMPTZ, bucket_label VARCHAR)'
    )
    self.db.executemany(
        'INSERT INTO _select_scores_buckets VALUES (?, ?)',
        [(edge[0], edge[1]) for edge in date_bins],
    )
    bucket_join = (
        ' ASOF JOIN _select_scores_buckets AS buckets'
        f' ON {datetime_column} >= buckets.bucket_start'
    )
    return bucket_join, 'buckets.bucket_label'

  def _build_stratum_expr(
      self,
      by_deployment_fields: Sequence[str],
      date_bins: Any,
      date_bin_expr: str,
      deployment_id_column: str = 'scores.deployment_id',
  ) -> str:
    """Build the stratum SQL expression by composing the strata axes.

    Shared by `select_scores()` and `select_detections()`.

    Args:
      by_deployment_fields: `deployments` column names to stratify by (e.g.
        `['id']` for the plain by-deployment case, or `['project',
        'management_unit']` to stratify by those `deployments` fields
        instead of/in addition to deployment identity). Empty if not
        stratifying by any deployment field.
      date_bins: Same as `select_scores()`'s `date_bins` argument (only
        used here to decide whether a `'date_bin'` key is also folded into
        the stratum).
      date_bin_expr: The date-bin SQL expression, from
        `_build_date_bin_clause()`.
      deployment_id_column: The (already-qualified) column referencing a
        row's deployment id -- substituted for the `'id'` field specifically
        (rather than re-referencing `deployments.id`), since callers already
        have this column available and it means the same thing.

    Returns:
      A SQL expression: a bare column reference if exactly one axis is
      active (one deployment field, xor one date bin, and nothing else), a
      `MAP`-literal-style dict if more than one axis is active, or the
      constant `0` if stratifying by nothing at all.
    """
    # `'id'`'s stratum key stays `'deployment_id'` (not `'id'`) for backwards
    # compatibility with the plain `by_deployment=True` case, which this
    # generalizes -- existing callers inspecting `stratum['deployment_id']`
    # must keep working unchanged.
    axes = {
        ('deployment_id' if field == 'id' else field): (
            deployment_id_column
            if field == 'id'
            else f'deployments.{quote_sql_identifier(field)}'
        )
        for field in by_deployment_fields
    }
    if date_bins is not None:
      axes['date_bin'] = date_bin_expr

    if not axes:
      return '0'
    if len(axes) == 1:
      return next(iter(axes.values()))
    entries = ', '.join(f"'{key}': {expr}" for key, expr in axes.items())
    return '{' + entries + '}'

  def _build_random_select_clause(
      self,
      selection: Literal['top', 'random', 'binned'],
      random_seed: float | None,
      key_columns: Sequence[str] = ('recording_id', 'start_time', 'end_time'),
  ) -> tuple[str, str]:
    """Build the `ORDER BY` expression and extra `SELECT` clause for `random`.

    Shared by `select_scores()` and `select_detections()`.

    Returns:
      A tuple `(order_expr, random_select)`: `order_expr` is what to
      `ORDER BY` within each stratum (`'score DESC'` for `'top'`/`'binned'`,
      `'random_key'` for `'random'`), and `random_select` is an extra
      `, ... AS random_key` clause to splice into the `SELECT` list (empty
      string unless `selection='random'`).
    """
    order_expr = 'random_key' if selection == 'random' else 'score DESC'
    if selection != 'random':
      return order_expr, ''

    if random_seed is not None:
      # `ORDER BY random()` is NOT reproducible in DuckDB across repeated
      # calls with the same `setseed()`: with a parallel query plan (e.g.
      # scanning more than one deployment's Parquet file), `random()`'s
      # global stream is consumed in a thread-scheduling-dependent order,
      # so the same seed can yield a different sample from run to run. A
      # deterministic hash of each row's own stable key + the seed has no
      # such dependency, since it is a pure function with no shared mutable
      # state, so it is used instead whenever reproducibility is requested.
      key_expr = " , '_', ".join(f'CAST({c} AS VARCHAR)' for c in key_columns)
      random_select = (
          f" , hash(concat({key_expr}, '_seed_{float(random_seed)}'))"
          ' AS random_key'
      )
    else:
      random_select = ', random() AS random_key'
    return order_expr, random_select

  def _normalize_by_deployment_fields(
      self,
      by_deployment: bool,
      by_deployment_fields: Sequence[str] | None,
  ) -> tuple[str, ...]:
    """Reconcile `by_deployment` (legacy) with `by_deployment_fields`.

    Shared by `select_scores()` and `select_detections()`. `by_deployment`
    is kept as a convenience boolean, equivalent to
    `by_deployment_fields=['id']`; passing both is rejected rather than
    silently picking one, since there's no obviously-correct way to combine
    a legacy flag with the field it's shorthand for.

    Args:
      by_deployment: The legacy boolean flag.
      by_deployment_fields: `deployments` column names to stratify by, or
        `None` for none.

    Returns:
      A tuple of `deployments` column names to stratify by (possibly
      empty), validated against the table's actual columns.

    Raises:
      ValueError: If both `by_deployment` and `by_deployment_fields` are
        given, or if `by_deployment_fields` names an unknown column.
    """
    if by_deployment and by_deployment_fields:
      raise ValueError(
          'Pass only one of `by_deployment` (shorthand for'
          " `by_deployment_fields=['id']`) or `by_deployment_fields`, not"
          ' both.'
      )
    fields = tuple(by_deployment_fields) if by_deployment_fields else (
        ('id',) if by_deployment else ()
    )
    self._validate_columns('deployments', fields)
    return fields

  def select_scores(
      self,
      class_name: str,
      selection: Literal['top', 'random', 'binned'] = 'top',
      n: int | None = None,
      bin_edges: Sequence[float] | None = None,
      by_deployment: bool = False,
      by_deployment_fields: Sequence[str] | None = None,
      date_bins: Literal['day', 'week']
      | Sequence[tuple[dt.datetime, str]]
      | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      min_score: float | None = None,
      max_score: float | None = None,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
      random_seed: float | None = None,
  ) -> Any:
    """Select scores for a single class in one DuckDB query, no Python loop.

    This is the general-purpose scores-selection entry point: it composes a
    selection mode (highest-scoring, random, or percentile-binned) with
    optional, independent stratification (by deployment, by arbitrary
    `deployments` field(s), by date bin, or any combination) and a set of
    filters, all evaluated in a single query over the relevant deployments'
    Parquet files.

    Args:
      class_name: The class to select scores for. Must be one of
        `classes`.
      selection: How to select rows (within each stratum, if `by_deployment`
        and/or `date_bins` is set; otherwise across the whole filtered
        result):
          - `'top'`: the `n` highest-scoring rows.
          - `'random'`: `n` uniformly random rows (see `random_seed`).
          - `'binned'`: `n` rows from *each* percentile bin defined by
            `bin_edges`, selected by highest score within the bin. Bin
            cutpoints are computed from the (stratum-local, if stratified)
            score distribution via `quantile_cont`, not fixed score values.
      n: Rows to select (per stratum and, for `'binned'`, per bin). Required
        for all selection modes.
      bin_edges: Percentile bin edges in `[0, 1]`, e.g. `[0, .5, .75, .875,
        1]` for 4 bins. Required (and only used) when `selection='binned'`.
      by_deployment: If `True`, `selection` is applied independently within
        each deployment, rather than once across all matching rows.
        Shorthand for `by_deployment_fields=['id']`; passing both raises.
        Composes with `date_bins` (all three can be set at once, modulo
        picking only one of `by_deployment`/`by_deployment_fields`).
      by_deployment_fields: If provided, `selection` is applied
        independently within each distinct combination of values of these
        `deployments` columns (built-in -- `id`, `name`, `project`,
        `latitude`, `longitude` -- or extra, e.g. a caller-added
        `management_unit` or `experimental_group` column). Each requested
        field also becomes its own output column (see `Returns` below), not
        just a `stratum` entry. `['id']` is equivalent to
        `by_deployment=True`; pass at most one of the two. Composes with
        `date_bins`.
      date_bins: If provided, `selection` is applied independently within
        each date bin. Either `'day'`/`'week'` (regular calendar buckets via
        `date_trunc`), or an explicit, caller-supplied sequence of
        `(bin_start, bin_label)` pairs (assigned via an as-of join, so each
        recording falls into the latest bin whose start is `<=` its
        `datetime`) -- each `bin_start` must be a timezone-aware datetime,
        same as `start_date`/`end_date` below. Composes with `by_deployment`/
        `by_deployment_fields`.
      deployments_filter: If provided, restricts to deployments matching
        these constraints. IMPORTANT: filter on the deployments table's own
        columns (`id`, `name`, `project`, `latitude`, `longitude`, or any
        extra column added via `add_extra_table_column('deployments',
        ...)`) -- e.g. `eq=dict(id=3)`, NOT `eq=dict(deployment_id=3)`. A
        deployment's own primary key is always called `id`;
        `deployment_id` only exists as a *foreign key* column on
        `recordings`/`detections` (and as a column name in this method's
        own output frame, which is a common source of confusion). An
        unknown column name raises a `ValueError` naming the valid columns.
      recordings_filter: If provided, restricts to recordings matching these
        constraints (columns: `id`, `filename`, `datetime`,
        `deployment_id`, or extras -- `deployment_id` IS correct here,
        since it is recordings' own foreign key column).
      min_score: If provided, only rows with `score >= min_score`.
      max_score: If provided, only rows with `score <= max_score`.
      start_date: If provided, only recordings with `datetime >= start_date`.
        Must be timezone-aware; a naive datetime is rejected (see
        `insert_recording`).
      end_date: If provided, only recordings with `datetime <= end_date`.
        Must be timezone-aware; a naive datetime is rejected.
      random_seed: Only used when `selection='random'`. If provided, makes
        the sample reproducible (same seed + same data -> same rows).

    Returns:
      A pandas DataFrame with columns `deployment_id`, `deployment_name`,
      `recording_id`, `recording_filename`, `recording_datetime`,
      `start_time`, `end_time`, `score`, plus one column per
      `by_deployment_fields` entry other than `'id'` (named after the
      field, e.g. a `project` column for `by_deployment_fields=['project']`
      -- `'id'`/`by_deployment` don't add a column of their own, since
      `deployment_id` already covers it), `stratum` (if stratifying by
      anything at all), `date_bin_label` (if `date_bins` is used), and
      `bin_idx` (if `selection='binned'`). Row order is by stratum/bin (if
      any), then by score descending.

    Example:
      A single call exercising every parameter: for class `'crow'`, within
      each `(deployment, week)` stratum, take the single highest-scoring
      row from each of 4 score-percentile bins, restricted to deployments
      in project `'my_project'`, recordings from a specific site
      (`site_id` extra column) recorded between two dates, with scores
      between 0.2 and 0.95:

      ```python
      import datetime as dt
      from ml_collections import config_dict

      result = db.select_scores(
          class_name='crow',
          selection='binned',
          n=1,
          bin_edges=[0, 0.5, 0.75, 0.875, 1],  # 4 percentile bins.
          by_deployment=True,
          date_bins='week',
          deployments_filter=config_dict.create(
              eq=dict(project='my_project'),  # NOT deployment_id=...; see above.
          ),
          recordings_filter=config_dict.create(
              eq=dict(site_id='SITE_042'),  # An extra column on recordings.
          ),
          min_score=0.2,
          max_score=0.95,
          start_date=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
          end_date=dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
          random_seed=0,  # Unused for selection='binned'; harmless to pass.
      )
      # result columns: deployment_id, deployment_name, recording_id,
      # recording_filename, recording_datetime, start_time, end_time,
      # score, stratum (a dict: {'deployment_id': ..., 'date_bin': ...}),
      # date_bin_label, bin_idx.
      ```

      Stratifying by an arbitrary `deployments` field instead of deployment
      identity, e.g. a caller-added `management_unit` extra column:

      ```python
      result = db.select_scores(
          class_name='crow',
          selection='top',
          n=5,
          by_deployment_fields=['management_unit'],
      )
      # result columns include a `management_unit` column directly (not
      # just inside `stratum`), so e.g. result.groupby('management_unit')
      # works without unpacking `stratum` first.
      ```
    """
    if n is None:
      raise ValueError('`n` is required.')
    if selection == 'binned' and not bin_edges:
      raise ValueError("`bin_edges` is required when selection='binned'.")
    require_timezone_aware(start_date, 'start_date')
    require_timezone_aware(end_date, 'end_date')
    if date_bins is not None and not isinstance(date_bins, str):
      for bin_start, _ in date_bins:
        require_timezone_aware(bin_start, 'date_bins')
    deployment_fields = self._normalize_by_deployment_fields(
        by_deployment, by_deployment_fields
    )
    stratify = bool(deployment_fields) or date_bins is not None
    # Validate the class name unconditionally (not just when a Parquet scan
    # actually happens), so an unknown class name is caught even when the
    # database has no deployments/scores yet.
    self._class_column(class_name)
    self._validate_filter_columns('recordings', recordings_filter)

    if deployments_filter is not None:
      deployment_ids = [
          d.id for d in self.get_all_deployments(filter=deployments_filter)
      ]
    else:
      deployment_ids = [d.id for d in self.get_all_deployments()]
    if not deployment_ids:
      return _empty_select_scores_frame(stratify, date_bins, selection)

    union_query = self._scores_union_query(class_name, deployment_ids)
    if union_query is None:
      return _empty_select_scores_frame(stratify, date_bins, selection)

    # Build the recordings-side filter (metadata filter + date range), which
    # is applied via a JOIN against `recordings` (dates and most recording
    # metadata don't exist in the Parquet files themselves).
    conditions, values = format_sql_where_conditions(
        recordings_filter, table_prefix='recordings'
    )
    date_conditions = []
    if start_date is not None:
      date_conditions.append('recordings.datetime >= ?')
      values.append(start_date)
    if end_date is not None:
      date_conditions.append('recordings.datetime <= ?')
      values.append(end_date)
    all_conditions = [c for c in [conditions] + date_conditions if c]
    if min_score is not None:
      all_conditions.append('scores.score >= ?')
      values.append(min_score)
    if max_score is not None:
      all_conditions.append('scores.score <= ?')
      values.append(max_score)
    where_clause = (
        f"WHERE {' AND '.join(all_conditions)}" if all_conditions else ''
    )

    # Build the date-bin expression (used for both stratification, when
    # `date_bins` contributes to `stratum`, and as its own output column).
    bucket_join, date_bin_expr = self._build_date_bin_clause(date_bins)

    # Build the stratum expression (or a constant, if not stratifying) by
    # independently composing `deployment_fields` and `date_bins`.
    stratum_expr = self._build_stratum_expr(
        deployment_fields, date_bins, date_bin_expr
    )
    # Fields other than `'id'` also become their own named output column
    # (not just a `stratum` entry) -- `'id'` doesn't need one, since
    # `deployment_id` already covers it.
    named_fields = [f for f in deployment_fields if f != 'id']
    field_columns_str = ''.join(
        f', deployments.{quote_sql_identifier(f)} AS {quote_sql_identifier(f)}'
        for f in named_fields
    )

    order_expr, random_select = self._build_random_select_clause(
        selection, random_seed
    )

    output_columns = [
        'deployment_id',
        'deployment_name',
        *named_fields,
        'recording_id',
        'recording_filename',
        'recording_datetime',
        'start_time',
        'end_time',
        'score',
    ]
    if stratify:
      output_columns.append('stratum')
    if date_bins is not None:
      output_columns.append('date_bin_label')

    try:
      base_query = f"""
          SELECT
              scores.deployment_id AS deployment_id,
              deployments.name AS deployment_name,
              scores.recording_id AS recording_id,
              recordings.filename AS recording_filename,
              recordings.datetime AS recording_datetime,
              scores.start_time AS start_time,
              scores.end_time AS end_time,
              scores.score AS score,
              {stratum_expr} AS stratum,
              {date_bin_expr} AS date_bin_label
              {field_columns_str}
              {random_select}
          FROM ({union_query}) AS scores
          JOIN recordings ON scores.recording_id = recordings.id
          JOIN deployments ON recordings.deployment_id = deployments.id
          {bucket_join}
          {where_clause}
          """

      if selection == 'binned':
        cutpoint_exprs = ', '.join(
            f'quantile_cont(score, {float(edge)}) OVER (PARTITION BY stratum)'
            f' AS cutpoint_{i}'
            for i, edge in enumerate(bin_edges)
        )
        bin_case = ' '.join(
            f'WHEN score <= cutpoint_{i + 1} THEN {i}'
            for i in range(len(bin_edges) - 1)
        )
        query = f"""
            WITH base AS ({base_query}),
            cutpoints AS (
                SELECT *, {cutpoint_exprs} FROM base
            ),
            binned AS (
                SELECT *, CASE {bin_case} END AS bin_idx FROM cutpoints
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY stratum, bin_idx ORDER BY score DESC
                ) AS rn
                FROM binned
                WHERE bin_idx IS NOT NULL
            )
            SELECT {', '.join(output_columns)}, bin_idx
            FROM ranked
            WHERE rn <= {int(n)}
            ORDER BY stratum, bin_idx, score DESC
            """
      else:
        query = f"""
            WITH base AS ({base_query}),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY stratum ORDER BY {order_expr}
                ) AS rn
                FROM base
            )
            SELECT {', '.join(output_columns)}
            FROM ranked
            WHERE rn <= {int(n)}
            ORDER BY stratum, score DESC
            """

      result = self.db.execute(query, values).fetchdf()
    finally:
      self.db.execute('DROP TABLE IF EXISTS _select_scores_buckets')

    return result

  def select_detections(
      self,
      class_name: str | None = None,
      selection: Literal['top', 'random', 'binned'] = 'top',
      n: int | None = None,
      bin_edges: Sequence[float] | None = None,
      by_deployment: bool = False,
      by_deployment_fields: Sequence[str] | None = None,
      date_bins: Literal['day', 'week']
      | Sequence[tuple[dt.datetime, str]]
      | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      min_score: float | None = None,
      max_score: float | None = None,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
      random_seed: float | None = None,
  ) -> Any:
    """Select detections in one DuckDB query, mirroring `select_scores()`.

    Composes a selection mode (highest-scoring, random, or
    percentile-binned) with optional, independent stratification (by
    deployment, by arbitrary `deployments` field(s), by date bin, or any
    combination) and a set of filters -- same shape as `select_scores()`,
    but reading from the `detections` table (a normal, small DuckDB table)
    rather than scanning per-deployment Parquet files, since detections are
    already sparse and id-bearing.

    Unlike `select_scores()`, `class_name` is optional: a detection already
    carries its own `class_name`, so omitting it selects across all classes
    rather than requiring one class per call.

    Args:
      class_name: If provided, restricts to detections for this class. Must
        be one of `classes`. If omitted, detections for any
        class are considered.
      selection: Same as `select_scores()`: `'top'` (the `n` highest-scoring
        detections), `'random'` (`n` uniformly random detections, see
        `random_seed`), or `'binned'` (`n` detections from each percentile
        bin defined by `bin_edges`).
      n: Detections to select (per stratum and, for `'binned'`, per bin).
        Required for all selection modes.
      bin_edges: Percentile bin edges in `[0, 1]`. Required (and only used)
        when `selection='binned'`.
      by_deployment: If `True`, `selection` is applied independently within
        each deployment. Shorthand for `by_deployment_fields=['id']`;
        passing both raises. Composes with `date_bins`.
      by_deployment_fields: Same as `select_scores()`'s
        `by_deployment_fields` -- stratify by arbitrary `deployments`
        column(s) instead of (or in addition to, via more than one field)
        deployment identity, each also becoming its own output column. At
        most one of `by_deployment`/`by_deployment_fields`. Composes with
        `date_bins`.
      date_bins: If provided, `selection` is applied independently within
        each date bin -- same semantics as `select_scores()`'s `date_bins`.
        Composes with `by_deployment`/`by_deployment_fields`.
      deployments_filter: If provided, restricts to deployments matching
        these constraints. IMPORTANT: use the deployments table's own
        columns (`id`, `name`, `project`, `latitude`, `longitude`, or
        extras) -- e.g. `eq=dict(id=3)`, NOT `eq=dict(deployment_id=3)` (see
        `select_scores()`'s docstring for why this is an easy mistake). An
        unknown column name raises a `ValueError` naming the valid columns.
      recordings_filter: If provided, restricts to recordings matching these
        constraints (columns: `id`, `filename`, `datetime`,
        `deployment_id`, or extras).
      filter: If provided, restricts to detections matching these
        constraints directly (columns: `id`, `recording_id`, `start_time`,
        `end_time`, `class_name`, `score`, or an extra column added via
        `add_extra_table_column('detections', ...)`, e.g.
        `eq=dict(provenance='auto')`).
      min_score: If provided, only detections with `score >= min_score`.
      max_score: If provided, only detections with `score <= max_score`.
      start_date: If provided, only recordings with `datetime >=
        start_date`. Must be timezone-aware; a naive datetime is rejected.
      end_date: If provided, only recordings with `datetime <= end_date`.
        Must be timezone-aware; a naive datetime is rejected.
      random_seed: Only used when `selection='random'`. If provided, makes
        the sample reproducible (same seed + same data -> same rows).

    Returns:
      A pandas DataFrame with columns `detection_id`, `deployment_id`,
      `deployment_name`, `recording_id`, `recording_filename`,
      `recording_datetime`, `start_time`, `end_time`, `class_name`, `score`,
      plus one column per `by_deployment_fields` entry other than `'id'`
      (see `select_scores()`'s `Returns` for details), `stratum` (if
      stratifying by anything at all), `date_bin_label` (if `date_bins` is
      used), and `bin_idx` (if `selection='binned'`). Row order is by
      stratum/bin (if any), then by score descending.

    Example:
      A single call exercising every parameter: among detections for class
      `'crow'` (added via `add_detections_from_threshold()` with
      `provenance='auto'`), within each `(deployment, week)` stratum, take
      the single highest-scoring detection from each of 4 score-percentile
      bins, restricted to deployments in project `'my_project'`, recordings
      from a specific site, detected between two dates, with scores between
      0.2 and 0.95:

      ```python
      import datetime as dt
      from ml_collections import config_dict

      result = db.select_detections(
          class_name='crow',
          selection='binned',
          n=1,
          bin_edges=[0, 0.5, 0.75, 0.875, 1],  # 4 percentile bins.
          by_deployment=True,
          date_bins='week',
          deployments_filter=config_dict.create(
              eq=dict(project='my_project'),  # NOT deployment_id=...; see above.
          ),
          recordings_filter=config_dict.create(
              eq=dict(site_id='SITE_042'),  # An extra column on recordings.
          ),
          filter=config_dict.create(
              eq=dict(provenance='auto'),  # An extra column on detections.
          ),
          min_score=0.2,
          max_score=0.95,
          start_date=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
          end_date=dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
          random_seed=0,  # Unused for selection='binned'; harmless to pass.
      )
      # result columns: detection_id, deployment_id, deployment_name,
      # recording_id, recording_filename, recording_datetime, start_time,
      # end_time, class_name, score, stratum, date_bin_label, bin_idx.
      ```
    """
    if n is None:
      raise ValueError('`n` is required.')
    if selection == 'binned' and not bin_edges:
      raise ValueError("`bin_edges` is required when selection='binned'.")
    require_timezone_aware(start_date, 'start_date')
    require_timezone_aware(end_date, 'end_date')
    if date_bins is not None and not isinstance(date_bins, str):
      for bin_start, _ in date_bins:
        require_timezone_aware(bin_start, 'date_bins')
    if class_name is not None:
      # Validates that `class_name` is one of `classes`; the
      # quoted-identifier return value isn't needed here since `class_name`
      # is bound as a query parameter (`detections.class_name = ?`), not
      # spliced into the query as a column reference.
      self._class_column(class_name)
    self._validate_filter_columns('detections', filter)
    self._validate_filter_columns('recordings', recordings_filter)
    self._validate_filter_columns('deployments', deployments_filter)
    deployment_fields = self._normalize_by_deployment_fields(
        by_deployment, by_deployment_fields
    )
    stratify = bool(deployment_fields) or date_bins is not None

    conditions, values = format_sql_where_conditions(
        filter, table_prefix='detections'
    )
    recordings_conditions, recordings_values = format_sql_where_conditions(
        recordings_filter, table_prefix='recordings'
    )
    deployments_conditions, deployments_values = format_sql_where_conditions(
        deployments_filter, table_prefix='deployments'
    )
    all_conditions = [
        c
        for c in [conditions, recordings_conditions, deployments_conditions]
        if c
    ]
    all_values = values + recordings_values + deployments_values
    if class_name is not None:
      all_conditions.append('detections.class_name = ?')
      all_values.append(class_name)
    if start_date is not None:
      all_conditions.append('recordings.datetime >= ?')
      all_values.append(start_date)
    if end_date is not None:
      all_conditions.append('recordings.datetime <= ?')
      all_values.append(end_date)
    if min_score is not None:
      all_conditions.append('detections.score >= ?')
      all_values.append(min_score)
    if max_score is not None:
      all_conditions.append('detections.score <= ?')
      all_values.append(max_score)
    where_clause = (
        f"WHERE {' AND '.join(all_conditions)}" if all_conditions else ''
    )

    bucket_join, date_bin_expr = self._build_date_bin_clause(date_bins)
    stratum_expr = self._build_stratum_expr(
        deployment_fields,
        date_bins,
        date_bin_expr,
        deployment_id_column='recordings.deployment_id',
    )
    named_fields = [f for f in deployment_fields if f != 'id']
    field_columns_str = ''.join(
        f', deployments.{quote_sql_identifier(f)} AS {quote_sql_identifier(f)}'
        for f in named_fields
    )
    order_expr, random_select = self._build_random_select_clause(
        selection,
        random_seed,
        key_columns=('detections.id',),
    )

    output_columns = [
        'detection_id',
        'deployment_id',
        'deployment_name',
        *named_fields,
        'recording_id',
        'recording_filename',
        'recording_datetime',
        'start_time',
        'end_time',
        'class_name',
        'score',
    ]
    if stratify:
      output_columns.append('stratum')
    if date_bins is not None:
      output_columns.append('date_bin_label')

    try:
      base_query = f"""
          SELECT
              detections.id AS detection_id,
              recordings.deployment_id AS deployment_id,
              deployments.name AS deployment_name,
              detections.recording_id AS recording_id,
              recordings.filename AS recording_filename,
              recordings.datetime AS recording_datetime,
              detections.start_time AS start_time,
              detections.end_time AS end_time,
              detections.class_name AS class_name,
              detections.score AS score,
              {stratum_expr} AS stratum,
              {date_bin_expr} AS date_bin_label
              {field_columns_str}
              {random_select}
          FROM detections
          JOIN recordings ON detections.recording_id = recordings.id
          JOIN deployments ON recordings.deployment_id = deployments.id
          {bucket_join}
          {where_clause}
          """

      if selection == 'binned':
        cutpoint_exprs = ', '.join(
            f'quantile_cont(score, {float(edge)}) OVER (PARTITION BY stratum)'
            f' AS cutpoint_{i}'
            for i, edge in enumerate(bin_edges)
        )
        bin_case = ' '.join(
            f'WHEN score <= cutpoint_{i + 1} THEN {i}'
            for i in range(len(bin_edges) - 1)
        )
        query = f"""
            WITH base AS ({base_query}),
            cutpoints AS (
                SELECT *, {cutpoint_exprs} FROM base
            ),
            binned AS (
                SELECT *, CASE {bin_case} END AS bin_idx FROM cutpoints
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY stratum, bin_idx ORDER BY score DESC
                ) AS rn
                FROM binned
                WHERE bin_idx IS NOT NULL
            )
            SELECT {', '.join(output_columns)}, bin_idx
            FROM ranked
            WHERE rn <= {int(n)}
            ORDER BY stratum, bin_idx, score DESC
            """
      else:
        query = f"""
            WITH base AS ({base_query}),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY stratum ORDER BY {order_expr}
                ) AS rn
                FROM base
            )
            SELECT {', '.join(output_columns)}
            FROM ranked
            WHERE rn <= {int(n)}
            ORDER BY stratum, score DESC
            """

      result = self.db.execute(query, all_values).fetchdf()
    finally:
      self.db.execute('DROP TABLE IF EXISTS _select_scores_buckets')

    return result

  def sampling_effort(
      self,
      by_deployment: bool = False,
      date_bins: Literal['day', 'week']
      | Sequence[tuple[dt.datetime, str]]
      | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
  ) -> Any:
    """Count scored windows per deployment x date stratum.

    Mirrors `select_scores()`'s stratification/filter arguments
    (`by_deployment`, `date_bins`, `deployments_filter`, `recordings_filter`,
    `start_date`, `end_date`), but reports *how much data exists* (a window
    count per stratum) rather than selecting score rows -- e.g. "how many
    scored clips does each deployment have per week" to check for coverage
    gaps, rather than "what are the highest scores per week."

    Unlike `select_scores()`/`select_detections()`, every stratum in the
    grid this method defines gets a row, even ones with zero windows: a
    stratum that has no matching data is not silently absent, it is a row
    with `n_windows=0`. The grid is built as follows:
      - The deployment axis (only present if `by_deployment=True`) is every
        deployment matching `deployments_filter`, independent of whether it
        has any recordings or scores yet.
      - The date-bin axis (only present if `date_bins` is not `None`)
        depends on the kind of `date_bins`:
          - An explicit `(bin_start, bin_label)` sequence defines its own
            fixed set of bins, independent of the data.
          - `'day'`/`'week'` (calendar truncation) has no such
            data-independent universe, so the bin axis is instead the
            *union*, across every deployment, of calendar bins touched by
            any recording matching `recordings_filter`/`start_date`/
            `end_date` -- e.g. with `date_bins='week'` and
            `by_deployment=True`, if deployment B has a recording in week 3
            but deployment A does not, deployment A still gets a
            `(A, week 3)` row with `n_windows=0`, surfacing the coverage
            gap. If no recording matches at all, there are no calendar bins
            to report and the result is empty (there is no
            data-independent notion of "which weeks exist").
      - If both axes are active, the grid is their full cross product
        (every matching deployment x every bin in the shared bin set). If
        neither is active, the result is a single row (the filtered
        database's total window count).

    Args:
      by_deployment: If `True`, stratify by deployment (see grid rules
        above). Composes with `date_bins`.
      date_bins: If provided, stratify by date bin -- see grid rules above
        for how the bin set is determined. Composes with `by_deployment`.
      deployments_filter: If provided, restricts the deployment axis (and
        the windows counted) to deployments matching these constraints.
        Use the deployments table's own columns (`id`, `name`, `project`,
        `latitude`, `longitude`, or extras) -- e.g. `eq=dict(id=3)`, NOT
        `eq=dict(deployment_id=3)` (see `select_scores()`'s docstring for
        why this is an easy mistake).
      recordings_filter: If provided, restricts the recordings (and
        therefore windows) counted to recordings matching these constraints
        (columns: `id`, `filename`, `datetime`, `deployment_id`, or
        extras).
      start_date: If provided, only recordings with `datetime >=
        start_date` are counted. Must be timezone-aware; a naive datetime
        is rejected.
      end_date: If provided, only recordings with `datetime <= end_date`
        are counted. Must be timezone-aware; a naive datetime is rejected.

    Returns:
      A pandas DataFrame with columns `n_windows`, plus `deployment_id` and
      `deployment_name` (if `by_deployment`), plus `date_bin_label` (if
      `date_bins` is set). One row per stratum in the grid described above,
      including strata with `n_windows=0`.

    Example:
      Windows per `(deployment, week)`, restricted to one project and a
      date range, to spot coverage gaps:

      ```python
      import datetime as dt
      from ml_collections import config_dict

      result = db.sampling_effort(
          by_deployment=True,
          date_bins='week',
          deployments_filter=config_dict.create(
              eq=dict(project='my_project'),
          ),
          start_date=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
          end_date=dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
      )
      # result columns: deployment_id, deployment_name, date_bin_label,
      # n_windows. One row per (deployment, week actually present in the
      # filtered data) -- including (deployment, week) pairs with
      # n_windows=0 if that deployment had no recordings that week but
      # some other matching deployment did.
      ```
    """
    require_timezone_aware(start_date, 'start_date')
    require_timezone_aware(end_date, 'end_date')
    if date_bins is not None and not isinstance(date_bins, str):
      for bin_start, _ in date_bins:
        require_timezone_aware(bin_start, 'date_bins')
    self._validate_filter_columns('recordings', recordings_filter)

    if deployments_filter is not None:
      deployments = self.get_all_deployments(filter=deployments_filter)
    else:
      deployments = self.get_all_deployments()
    deployment_ids = [d.id for d in deployments]
    if by_deployment and not deployments:
      return _empty_sampling_effort_frame(by_deployment, date_bins)

    # Recordings-side filter (metadata filter + date range + the
    # deployments_filter-scoped deployment set, if stratifying by
    # deployment), shared by the bin-set query and the window-count query
    # below.
    conditions, values = format_sql_where_conditions(
        recordings_filter, table_prefix='recordings'
    )
    date_conditions = []
    if start_date is not None:
      date_conditions.append('recordings.datetime >= ?')
      values.append(start_date)
    if end_date is not None:
      date_conditions.append('recordings.datetime <= ?')
      values.append(end_date)
    if by_deployment:
      # Scope counted recordings to the (already deployments_filter-scoped)
      # deployment set, so e.g. a deployments_filter restricting to one
      # project doesn't pull in other projects' recordings when computing
      # the shared calendar bin set below.
      placeholders = ', '.join(['?'] * len(deployment_ids)) or 'NULL'
      date_conditions.append(f'recordings.deployment_id IN ({placeholders})')
      values.extend(deployment_ids)
    all_conditions = [c for c in [conditions] + date_conditions if c]
    where_clause = (
        f"WHERE {' AND '.join(all_conditions)}" if all_conditions else ''
    )
    recordings_values = values

    try:
      bucket_join, date_bin_expr = self._build_date_bin_clause(date_bins)

      # Build the bin axis, if stratifying by date. See the docstring for
      # why calendar bins ('day'/'week') and an explicit bin list have
      # different universes.
      bin_labels = None
      if date_bins is not None:
        if isinstance(date_bins, str):
          bin_rows = self.db.execute(
              f"""
              SELECT DISTINCT {date_bin_expr} AS date_bin_label
              FROM recordings
              {bucket_join}
              {where_clause}
              """,
              recordings_values,
          ).fetchall()
          bin_labels = [r[0] for r in bin_rows if r[0] is not None]
        else:
          bin_labels = [label for _, label in date_bins]
        if not bin_labels:
          return _empty_sampling_effort_frame(by_deployment, date_bins)

      # Build the grid as a temp table with exactly the active axis
      # columns, independent of whether any scores/windows actually exist
      # for a given combination -- that's the whole point, see the
      # docstring. `by_deployment` and `date_bins` are each optional, so the
      # grid is the cross product of whichever axes are active: a list of
      # per-row dicts for the deployment axis (one dict per deployment, or a
      # single empty dict if not stratifying by deployment) crossed with a
      # list of per-row dicts for the date-bin axis (one per bin, or a
      # single empty dict if not stratifying by date), merging the two dicts
      # for every combination.
      self.db.execute('DROP TABLE IF EXISTS _sampling_effort_grid')
      grid_columns = []
      if by_deployment:
        grid_columns += ['deployment_id INTEGER', 'deployment_name VARCHAR']
      if date_bins is not None:
        grid_columns.append('date_bin_label VARCHAR')
      column_names = [c.split(' ')[0] for c in grid_columns]
      self.db.execute(
          'CREATE TEMP TABLE _sampling_effort_grid'
          f" ({', '.join(grid_columns) if grid_columns else 'placeholder INTEGER'})"
      )

      deployment_axis_rows = (
          [{'deployment_id': d.id, 'deployment_name': d.name} for d in deployments]
          if by_deployment
          else [{}]
      )
      date_bin_axis_rows = (
          [{'date_bin_label': label} for label in bin_labels]
          if date_bins is not None
          else [{}]
      )
      grid_rows = [
          tuple((dep_row | bin_row)[c] for c in column_names)
          for dep_row in deployment_axis_rows
          for bin_row in date_bin_axis_rows
      ] if column_names else [(1,)]
      placeholders_str = ', '.join(['?'] * max(len(grid_columns), 1))
      self.db.executemany(
          f'INSERT INTO _sampling_effort_grid VALUES ({placeholders_str})',
          grid_rows,
      )

      # Count actual windows per (deployment, date bin) touched by any
      # score chunk file, using the same recordings-side filter as the grid.
      union_query = self._windows_union_query(deployment_ids)
      count_select = ['COUNT(*) AS n_windows']
      count_group_by = []
      if by_deployment:
        count_select.append('scores.deployment_id AS deployment_id')
        count_group_by.append('scores.deployment_id')
      if date_bins is not None:
        count_select.append(f'{date_bin_expr} AS date_bin_label')
        count_group_by.append(date_bin_expr)
      group_by_clause = (
          f"GROUP BY {', '.join(count_group_by)}" if count_group_by else ''
      )
      if union_query is not None:
        counts_query = f"""
            SELECT {', '.join(count_select)}
            FROM ({union_query}) AS scores
            JOIN recordings ON scores.recording_id = recordings.id
            {bucket_join}
            {where_clause}
            {group_by_clause}
            """
        counts_values = recordings_values
      else:
        # No deployment in scope has any score chunk files at all -- select
        # zero rows, but with the right output column names/types (no
        # `scores` table exists in this branch to alias them from).
        empty_select = ['CAST(NULL AS BIGINT) AS n_windows']
        if by_deployment:
          empty_select.append('CAST(NULL AS INTEGER) AS deployment_id')
        if date_bins is not None:
          empty_select.append('CAST(NULL AS VARCHAR) AS date_bin_label')
        counts_query = f"SELECT {', '.join(empty_select)} WHERE FALSE"
        counts_values = []

      join_conditions = []
      if by_deployment:
        join_conditions.append('grid.deployment_id = counts.deployment_id')
      if date_bins is not None:
        join_conditions.append('grid.date_bin_label = counts.date_bin_label')
      join_clause = (
          f"ON {' AND '.join(join_conditions)}"
          if join_conditions
          else 'ON TRUE'
      )

      output_columns = []
      if by_deployment:
        output_columns += ['grid.deployment_id AS deployment_id',
                            'grid.deployment_name AS deployment_name']
      if date_bins is not None:
        output_columns.append('grid.date_bin_label AS date_bin_label')
      output_columns.append('COALESCE(counts.n_windows, 0) AS n_windows')
      order_by_clause = (
          f"ORDER BY {', '.join(c.split(' AS ')[1] for c in output_columns[:-1])}"
          if output_columns[:-1]
          else ''
      )

      final_query = f"""
          SELECT {', '.join(output_columns)}
          FROM _sampling_effort_grid AS grid
          LEFT JOIN ({counts_query}) AS counts
          {join_clause}
          {order_by_clause}
          """
      result = self.db.execute(final_query, counts_values).fetchdf()
    finally:
      self.db.execute('DROP TABLE IF EXISTS _select_scores_buckets')
      self.db.execute('DROP TABLE IF EXISTS _sampling_effort_grid')

    return result

  def get_scores_batch(
      self,
      deployment_id: int,
      recording_ids: Sequence[int],
      start_times: Sequence[float],
      end_times: Sequence[float],
  ) -> np.ndarray:
    """Get the full class-score vector for a batch of specific windows.

    Args:
      deployment_id: The deployment whose score chunk file(s) to read from.
        All requested windows must belong to this deployment.
      recording_ids: The recording id for each requested window.
      start_times: The start offset (seconds) for each requested window.
      end_times: The end offset (seconds) for each requested window.

    Returns:
      A `[len(recording_ids), len(classes)]` array of per-class
      scores, in the same order as the input keys and the same column order
      as `classes`.

    Raises:
      KeyError: If any requested `(recording_id, start_time, end_time)` key
        is not found in any of the deployment's score chunk files.
    """
    deployment_id = int(deployment_id)
    class_labels = self.classes
    n = len(recording_ids)
    if not (len(start_times) == n and len(end_times) == n):
      raise ValueError(
          '`recording_ids`, `start_times`, and `end_times` must all have the'
          f' same length (got {n}, {len(start_times)}, {len(end_times)}).'
      )

    # Resolve only the chunk files that actually contain a requested
    # recording, rather than every chunk file in the deployment -- a real
    # pruning win when the request is scoped to a handful of recordings out
    # of a much larger deployment (see PARQUET_DUCKDB_PLAN.md, Q9).
    unique_recording_ids = sorted(set(int(r) for r in recording_ids))
    placeholders = ', '.join(['?'] * len(unique_recording_ids))
    relevant_chunk_rows = self.db.execute(
        'SELECT DISTINCT score_chunk_path FROM recordings'
        f' WHERE id IN ({placeholders}) AND deployment_id = ?'
        ' AND score_chunk_path IS NOT NULL',
        [*unique_recording_ids, deployment_id],
    ).fetchall()
    chunk_paths = [row[0] for row in relevant_chunk_rows]
    if not chunk_paths:
      raise KeyError(
          'No score chunk files found for deployment'
          f' {deployment_id} covering the requested recordings.'
      )

    keys_table = self.db.from_arrow(
        _build_keys_arrow_table(recording_ids, start_times, end_times)
    )
    keys_table.create('_score_keys')
    try:
      class_columns_str = ', '.join(
          f'scores.{quote_sql_identifier(c)}' for c in class_labels
      )
      paths_str = ', '.join(f"'{p}'" for p in chunk_paths)
      query = f"""
          SELECT keys._req_idx, {class_columns_str}
          FROM _score_keys AS keys
          LEFT JOIN read_parquet([{paths_str}]) AS scores
          ON keys.recording_id = scores.recording_id
          AND ABS(keys.start_time - scores.start_time) < 1e-6
          AND ABS(keys.end_time - scores.end_time) < 1e-6
          ORDER BY keys._req_idx
          """
      rows = self.db.execute(query).fetchall()
    finally:
      self.db.execute('DROP TABLE IF EXISTS _score_keys')

    result = np.full((n, len(class_labels)), np.nan, dtype=np.float16)
    missing = []
    for row in rows:
      idx = row[0]
      values = row[1:]
      if values[0] is None:
        missing.append(idx)
        continue
      result[idx] = values
    if missing:
      raise KeyError(
          'No score row found for the following requested indices'
          f' (0-indexed into the input batch): {missing}.'
      )
    return result

  def get_scores(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      start_date: dt.datetime | None = None,
      end_date: dt.datetime | None = None,
  ) -> ScoreBatch:
    """Get all scored windows matching a filter, as a `ScoreBatch`.

    The columnar (struct-of-arrays), filtered-scan analog of
    `SQLiteUSearchDB.get_all_windows()`, adapted to this backend's wide,
    per-deployment-Parquet score storage: rather than a list of per-window
    objects, this returns one `ScoreBatch` -- parallel `deployment_ids`/
    `recording_ids`/`start_times`/`end_times` arrays plus a single `[N,
    n_classes]` `scores` array -- matching the shape
    `insert_deployment_scores()`/`get_scores_batch()` already use elsewhere
    in this class. With no arguments, returns every scored window in the
    database; each filter narrows the scan.

    Args:
      deployments_filter: If provided, restricts to deployments matching
        these constraints. Use the deployments table's own columns (`id`,
        `name`, `project`, `latitude`, `longitude`, or extras) -- e.g.
        `eq=dict(name='site_1')`, NOT `eq=dict(deployment_id='site_1')` (see
        `select_scores()`'s docstring for why this is an easy mistake).
      recordings_filter: If provided, restricts to recordings matching
        these constraints (columns: `id`, `filename`, `datetime`,
        `deployment_id`, or extras).
      start_date: If provided, only recordings with `datetime >=
        start_date`. Must be timezone-aware; a naive datetime is rejected.
      end_date: If provided, only recordings with `datetime <= end_date`.
        Must be timezone-aware; a naive datetime is rejected.

    Returns:
      A `ScoreBatch` with one entry per matching window (empty arrays, not
      an error, if nothing matches).

    Example:
      Scores for a specific list of deployments, by name:

      ```python
      from ml_collections import config_dict

      batch = db.get_scores(
          deployments_filter=config_dict.create(
              isin=dict(name=['site_1', 'site_2', 'site_7'])
          ),
      )
      ```

      Scores for a specific list of recordings, by filename:

      ```python
      batch = db.get_scores(
          recordings_filter=config_dict.create(
              isin=dict(filename=['clip_0001.wav', 'clip_0002.wav'])
          ),
      )
      ```

      Scores for a date range, across all deployments:

      ```python
      import datetime as dt

      batch = db.get_scores(
          start_date=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
          end_date=dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
      )
      ```

      Any of `deployments_filter`/`recordings_filter`/`start_date`/
      `end_date` can be combined in one call, e.g. one deployment within a
      date range. `batch.scores` is `[len(batch), len(batch.class_labels)]`;
      `batch.class_labels[j]` names `batch.scores[:, j]`'s class.
    """
    require_timezone_aware(start_date, 'start_date')
    require_timezone_aware(end_date, 'end_date')
    self._validate_filter_columns('recordings', recordings_filter)
    class_labels = self.classes

    if deployments_filter is not None:
      deployment_ids = [
          d.id for d in self.get_all_deployments(filter=deployments_filter)
      ]
    else:
      deployment_ids = [d.id for d in self.get_all_deployments()]
    if not deployment_ids:
      return _empty_score_batch(class_labels)

    union_query = self._wide_scores_union_query(deployment_ids)
    if union_query is None:
      return _empty_score_batch(class_labels)

    conditions, values = format_sql_where_conditions(
        recordings_filter, table_prefix='recordings'
    )
    date_conditions = []
    if start_date is not None:
      date_conditions.append('recordings.datetime >= ?')
      values.append(start_date)
    if end_date is not None:
      date_conditions.append('recordings.datetime <= ?')
      values.append(end_date)
    all_conditions = [c for c in [conditions] + date_conditions if c]
    where_clause = (
        f"WHERE {' AND '.join(all_conditions)}" if all_conditions else ''
    )

    class_columns_str = ', '.join(
        f'scores.{quote_sql_identifier(c)}' for c in class_labels
    )
    query = f"""
        SELECT
            scores.deployment_id,
            scores.recording_id,
            scores.start_time,
            scores.end_time,
            {class_columns_str}
        FROM ({union_query}) AS scores
        JOIN recordings ON scores.recording_id = recordings.id
        {where_clause}
        """
    rows = self.db.execute(query, values).fetchall()
    if not rows:
      return _empty_score_batch(class_labels)

    columns = np.array(rows, dtype=object)
    return ScoreBatch(
        deployment_ids=columns[:, 0].astype(np.int64),
        recording_ids=columns[:, 1].astype(np.int64),
        start_times=columns[:, 2].astype(np.float64),
        end_times=columns[:, 3].astype(np.float64),
        scores=columns[:, 4:].astype(np.float16),
        class_labels=class_labels,
    )

  def add_detections_from_threshold(
      self,
      threshold: float,
      class_name: str | None = None,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      handle_duplicates: Literal['overwrite', 'skip', 'error'] = 'overwrite',
      **extra_kwargs: Any,
  ) -> Sequence[int]:
    """Scan Parquet score files and insert qualifying detections.

    Runs a DuckDB query directly over the relevant deployments' Parquet
    files, selecting rows where `score >= threshold`, and bulk-inserts the
    results into the `detections` table.

    Args:
      threshold: The minimum score (inclusive) for a row to become a
        detection.
      class_name: If provided, restricts the scan to this class. Must be
        one of `classes`. If omitted, all classes are scanned (via a single
        `UNPIVOT`-based query, not one query per class), so any class
        scoring above `threshold` for a given window becomes a detection --
        note this means a single window can produce multiple detections
        (one per qualifying class) when `class_name` is omitted.
      deployments_filter: If provided, restricts the scan to deployments
        matching these constraints.
      recordings_filter: If provided, restricts the scan to recordings
        matching these constraints.
      handle_duplicates: Same as `insert_detection()`'s `handle_duplicates`,
        applied to every detection this call inserts -- e.g. re-running
        this method with the same arguments defaults to updating each
        matching detection's score in place, rather than creating
        duplicate rows.
      **extra_kwargs: Additional keyword arguments applied to every inserted
        detection (e.g. provenance/model-run metadata), via the same
        extra-column mechanism as `insert_detection()`.

    Returns:
      The ids of the inserted (or, for a duplicate match with
      `handle_duplicates='overwrite'`/`'skip'`, updated/existing)
      detections.
    """
    if deployments_filter is not None:
      deployment_ids = [
          d.id for d in self.get_all_deployments(filter=deployments_filter)
      ]
    else:
      deployment_ids = [d.id for d in self.get_all_deployments()]

    class_name_values = []
    if class_name is not None:
      self._class_column(class_name)
      union_query = self._scores_union_query(class_name, deployment_ids)
      class_name_select = '?'
      class_name_values = [class_name]
    else:
      union_query = self._scores_union_query_all_classes(deployment_ids)
      class_name_select = 'scores.class_name'
    if union_query is None:
      return []

    conditions, values = format_sql_where_conditions(
        recordings_filter, table_prefix='recordings'
    )
    conditions = f'{conditions} AND ' if conditions else ''
    query = f"""
        SELECT
            scores.deployment_id, scores.recording_id, scores.start_time,
            scores.end_time, {class_name_select} AS class_name, scores.score
        FROM ({union_query}) AS scores
        JOIN recordings ON scores.recording_id = recordings.id
        WHERE {conditions}scores.score >= ?
        """
    scores_frame = self.db.execute(
        query, class_name_values + values + [threshold]
    ).fetchdf()
    detections_batch = [
        dict(
            recording_id=int(row.recording_id),
            offsets=[float(row.start_time), float(row.end_time)],
            class_name=row.class_name,
            score=float(row.score),
            **extra_kwargs,
        )
        for row in scores_frame.itertuples()
    ]
    return self.insert_detections_batch(
        detections_batch, handle_duplicates=handle_duplicates
    )


def _build_scores_arrow_table(
    recording_ids: Sequence[int],
    start_times: Sequence[float],
    end_times: Sequence[float],
    scores: np.ndarray,
    class_labels: Sequence[str],
) -> Any:
  """Build a pyarrow Table for a deployment's score Parquet file."""
  import pyarrow as pa  # pylint: disable=g-import-not-at-top

  arrays = [
      pa.array(recording_ids, type=pa.int64()),
      pa.array(start_times, type=pa.float64()),
      pa.array(end_times, type=pa.float64()),
  ]
  names = ['recording_id', 'start_time', 'end_time']
  for i, class_name in enumerate(class_labels):
    arrays.append(pa.array(scores[:, i], type=pa.float16()))
    names.append(class_name)
  return pa.Table.from_arrays(arrays, names=names)


def _build_keys_arrow_table(
    recording_ids: Sequence[int],
    start_times: Sequence[float],
    end_times: Sequence[float],
) -> Any:
  """Build a pyarrow Table of lookup keys for `get_scores_batch()`."""
  import pyarrow as pa  # pylint: disable=g-import-not-at-top

  return pa.Table.from_arrays(
      [
          pa.array(range(len(recording_ids)), type=pa.int64()),
          pa.array(recording_ids, type=pa.int64()),
          pa.array(start_times, type=pa.float64()),
          pa.array(end_times, type=pa.float64()),
      ],
      names=['_req_idx', 'recording_id', 'start_time', 'end_time'],
  )


def _empty_select_scores_frame(
    stratify: bool, date_bins: Any, selection: str
) -> Any:
  """An empty DataFrame with the schema `select_scores()` would return."""
  import pandas as pd  # pylint: disable=g-import-not-at-top

  columns = [
      'deployment_id',
      'deployment_name',
      'recording_id',
      'recording_filename',
      'recording_datetime',
      'start_time',
      'end_time',
      'score',
  ]
  if stratify:
    columns.append('stratum')
  if date_bins is not None:
    columns.append('date_bin_label')
  if selection == 'binned':
    columns.append('bin_idx')
  return pd.DataFrame(columns=columns)


def _empty_sampling_effort_frame(by_deployment: bool, date_bins: Any) -> Any:
  """An empty DataFrame with the schema `sampling_effort()` would return."""
  import pandas as pd  # pylint: disable=g-import-not-at-top

  columns = []
  if by_deployment:
    columns += ['deployment_id', 'deployment_name']
  if date_bins is not None:
    columns.append('date_bin_label')
  columns.append('n_windows')
  return pd.DataFrame(columns=columns)


def _empty_score_batch(class_labels: Sequence[str]) -> ScoreBatch:
  """An empty `ScoreBatch` with the right dtypes/shape for `get_scores()`."""
  return ScoreBatch(
      deployment_ids=np.zeros((0,), dtype=np.int64),
      recording_ids=np.zeros((0,), dtype=np.int64),
      start_times=np.zeros((0,), dtype=np.float64),
      end_times=np.zeros((0,), dtype=np.float64),
      scores=np.zeros((0, len(class_labels)), dtype=np.float16),
      class_labels=class_labels,
  )
