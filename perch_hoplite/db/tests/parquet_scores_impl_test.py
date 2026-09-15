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

"""Tests for `parquet_scores_impl.ParquetScoresDB`.

This backend is not exercised by `hoplite_test.py`/`interface_test.py`'s
`DB_TYPES`-parametrized conformance suite: those tests assume every backend
has a `windows` table and embedding-vector storage/search, neither of which
`ParquetScoresDB` has (see `PARQUET_DUCKDB_PLAN.md`). This file exercises the
parts of `HopliteDBInterface` this backend does implement, plus its
backend-specific score/detection APIs.
"""

import datetime as dt
import shutil
import tempfile

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import parquet_scores_impl as psi

from absl.testing import absltest
from absl.testing import parameterized

CLASS_LABELS = ('alpha', 'beta', 'gamma', 'American Redstart', 'a-b')


class ParquetScoresDBTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def make_db(self, class_labels=CLASS_LABELS, readonly=False):
    return psi.ParquetScoresDB.create(
        db_path=self.tempdir, class_labels=class_labels, readonly=readonly
    )

  def insert_deployment_with_recordings(
      self, db, num_recordings=1, name='d', **kwargs
  ):
    deployment_id = db.insert_deployment(name=name, project='p', **kwargs)
    recording_ids = [
        db.insert_recording(filename=f'f{i}.wav', deployment_id=deployment_id)
        for i in range(num_recordings)
    ]
    return deployment_id, recording_ids

  # ---------------------------------------------------------------------
  # Creation / class vocabulary.
  # ---------------------------------------------------------------------

  def test_create_requires_class_labels_on_first_creation(self):
    with self.assertRaises(ValueError):
      psi.ParquetScoresDB.create(db_path=self.tempdir)

  def test_create_persists_and_reloads_class_labels(self):
    db = self.make_db()
    self.assertEqual(db.classes, CLASS_LABELS)
    db.commit()
    db.db.close()

    reloaded = psi.ParquetScoresDB.create(db_path=self.tempdir)
    self.assertEqual(reloaded.classes, CLASS_LABELS)

  def test_create_rejects_mismatched_class_labels(self):
    db = self.make_db()
    db.commit()
    db.db.close()

    with self.assertRaises(ValueError):
      psi.ParquetScoresDB.create(
          db_path=self.tempdir, class_labels=('different', 'labels')
      )

  def test_readonly_blocks_writes(self):
    # DuckDB refuses a second same-process connection (read-only or not) to
    # a file that already has an open connection, so the writable
    # connection must be closed before a read-only one can be opened.
    db = self.make_db()
    db.commit()
    db.db.close()

    ro_db = self.make_db(readonly=True)
    with self.assertRaises(Exception):
      ro_db.insert_deployment(name='d2', project='p')

  def test_thread_split_shares_connection_state(self):
    db = self.make_db()
    db.insert_deployment(name='d', project='p')
    db.commit()

    split = db.thread_split()
    self.assertLen(split.get_all_deployments(), 1)
    # The split shares the same underlying connection, so writes through it
    # are visible from the original instance too.
    split.insert_deployment(name='d2', project='p')
    self.assertLen(db.get_all_deployments(), 2)

  def test_thread_split_preserves_utc_timezone_pin(self):
    # Regression test: DuckDB's cursor() starts a fresh session state, so
    # session-level settings like the UTC TimeZone pin set in create() are
    # NOT inherited automatically -- thread_split() must reapply it, or
    # datetime handling on the split copy would silently depend on the
    # host's local timezone instead.
    db = self.make_db()
    deployment_id, _ = self.insert_deployment_with_recordings(db)
    recording_id = db.insert_recording(
        filename='dated.wav',
        deployment_id=deployment_id,
        datetime=dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )
    db.commit()

    split = db.thread_split()
    self.assertEqual(
        split.get_recording(recording_id).datetime,
        dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )

  # ---------------------------------------------------------------------
  # deployments / recordings.
  # ---------------------------------------------------------------------

  def test_insert_and_get_deployment(self):
    db = self.make_db()
    deployment_id = db.insert_deployment(
        name='d1', project='p1', latitude=1.0, longitude=2.0
    )
    deployment = db.get_deployment(deployment_id)
    self.assertEqual(deployment.name, 'd1')
    self.assertEqual(deployment.project, 'p1')
    self.assertEqual(deployment.latitude, 1.0)
    self.assertEqual(deployment.longitude, 2.0)

  def test_get_deployment_missing_raises_keyerror(self):
    db = self.make_db()
    with self.assertRaises(KeyError):
      db.get_deployment(12345)

  def test_deployment_extra_columns(self):
    db = self.make_db()
    deployment_id = db.insert_deployment(name='d1', project='p1', extra=42)
    self.assertEqual(
        db.get_extra_table_columns()['deployments'], {'extra': int}
    )
    self.assertEqual(db.get_deployment(deployment_id).extra, 42)

  def test_extra_datetime_column_naive_value_raises(self):
    # Regression test: a naive datetime passed via **kwargs into an extra,
    # caller-added dt.datetime-typed column (not just the built-in
    # `recordings.datetime`) must also be rejected -- the check lives in
    # normalize_sql_value(), the single choke point every inserted/filtered
    # value passes through, precisely so it isn't bypassable via kwargs.
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.insert_deployment(
          name='d1', project='p1', install_date=dt.datetime(2024, 1, 1)
      )

  def test_extra_datetime_column_tz_aware_value_round_trips(self):
    db = self.make_db()
    install_date = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    deployment_id = db.insert_deployment(
        name='d1', project='p1', install_date=install_date
    )
    self.assertEqual(
        db.get_deployment(deployment_id).install_date, install_date
    )

  def test_insert_recording_requires_valid_deployment(self):
    db = self.make_db()
    with self.assertRaises(RuntimeError):
      db.insert_recording(filename='f.wav', deployment_id=999)

  def test_insert_deployment_idempotent_after_it_has_recordings(self):
    # Regression test: DuckDB refuses `INSERT ... ON CONFLICT ... DO UPDATE`
    # against a row already referenced by a foreign key elsewhere, even for
    # a no-op update -- which fired on every idempotent re-run of an
    # ingestion script once a deployment had any recordings.
    db = self.make_db()
    deployment_id = db.insert_deployment(name='point_a', project='p1', latitude=1.0)
    db.insert_recording(filename='f1.wav', deployment_id=deployment_id)

    same_id = db.insert_deployment(name='point_a', project='p1', latitude=2.0)
    self.assertEqual(same_id, deployment_id)
    self.assertEqual(db.get_deployment(deployment_id).latitude, 2.0)

  def test_insert_deployment_extra_column_update_after_it_has_recordings(self):
    db = self.make_db()
    deployment_id = db.insert_deployment(name='point_a', project='p1')
    db.insert_recording(filename='f1.wav', deployment_id=deployment_id)

    same_id = db.insert_deployment(name='point_a', project='p1', extra=99)
    self.assertEqual(same_id, deployment_id)
    self.assertEqual(db.get_deployment(deployment_id).extra, 99)

  def test_insert_deployment_omitted_fields_preserve_existing_values(self):
    # Regression test: a duplicate-matching insert_deployment() call that
    # omits latitude/longitude (they default to None) must NOT blank out
    # previously-set values -- only an explicitly-passed value overwrites.
    db = self.make_db()
    deployment_id = db.insert_deployment(
        name='point_a', project='p1', latitude=5.0, longitude=6.0
    )

    same_id = db.insert_deployment(name='point_a', project='p1')
    self.assertEqual(same_id, deployment_id)
    deployment = db.get_deployment(deployment_id)
    self.assertEqual(deployment.latitude, 5.0)
    self.assertEqual(deployment.longitude, 6.0)

    # An explicitly-passed value still overwrites, and other omitted fields
    # remain preserved.
    db.insert_deployment(name='point_a', project='p1', latitude=9.0)
    deployment = db.get_deployment(deployment_id)
    self.assertEqual(deployment.latitude, 9.0)
    self.assertEqual(deployment.longitude, 6.0)

  def test_insert_deployment_omitted_fields_preserved_after_it_has_recordings(
      self,
  ):
    # Same as above, but via the FK-referenced-row fallback path (see
    # test_insert_deployment_idempotent_after_it_has_recordings) rather
    # than the fast-path upsert -- both paths must apply the same
    # preserve-existing-value semantics.
    db = self.make_db()
    deployment_id = db.insert_deployment(
        name='point_a', project='p1', latitude=5.0, longitude=6.0
    )
    db.insert_recording(filename='f1.wav', deployment_id=deployment_id)

    db.insert_deployment(name='point_a', project='p1')
    deployment = db.get_deployment(deployment_id)
    self.assertEqual(deployment.latitude, 5.0)
    self.assertEqual(deployment.longitude, 6.0)

  def test_insert_recording_omitted_datetime_preserves_existing_value(self):
    db = self.make_db()
    deployment_id, _ = self.insert_deployment_with_recordings(db)
    recording_id = db.insert_recording(
        filename='dated.wav',
        deployment_id=deployment_id,
        datetime=dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )

    same_id = db.insert_recording(
        filename='dated.wav', deployment_id=deployment_id
    )
    self.assertEqual(same_id, recording_id)
    self.assertEqual(
        db.get_recording(recording_id).datetime,
        dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )

  def test_recording_datetime_is_native_datetime_type(self):
    # The `recordings.datetime` column is a timezone-aware TIMESTAMPTZ (see
    # the docstring on `create()`'s UTC session-timezone pin), so a naive
    # input datetime round-trips as UTC-aware, not naive.
    db = self.make_db()
    deployment_id, _ = self.insert_deployment_with_recordings(db)
    recording_id = db.insert_recording(
        filename='dated.wav',
        deployment_id=deployment_id,
        datetime=dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )
    recording = db.get_recording(recording_id)
    self.assertIsInstance(recording.datetime, dt.datetime)
    self.assertIsNotNone(recording.datetime.tzinfo)
    self.assertEqual(
        recording.datetime, dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc)
    )

    [from_get_all] = [
        r for r in db.get_all_recordings() if r.id == recording_id
    ]
    self.assertIsInstance(from_get_all.datetime, dt.datetime)

  def test_recording_datetime_preserves_non_utc_timezone_aware_input(self):
    # Regression test: a tz-aware datetime in a non-UTC zone must round-trip
    # to the same instant, not be silently converted to the host's local
    # timezone and stored naive (DuckDB's plain TIMESTAMP behavior, which is
    # why this column uses TIMESTAMPTZ instead -- see create()'s docstring).
    db = self.make_db()
    deployment_id, _ = self.insert_deployment_with_recordings(db)
    mst = dt.timezone(dt.timedelta(hours=-7))
    input_dt = dt.datetime(2024, 1, 1, 3, 0, tzinfo=mst)  # 10:00 UTC.
    recording_id = db.insert_recording(
        filename='dated.wav', deployment_id=deployment_id, datetime=input_dt
    )
    recording = db.get_recording(recording_id)
    self.assertEqual(recording.datetime, input_dt)  # Same instant.
    self.assertEqual(
        recording.datetime.astimezone(dt.timezone.utc),
        dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc),
    )

  def test_insert_recording_naive_datetime_raises(self):
    # A naive (no tzinfo) datetime is rejected outright rather than silently
    # assumed to be UTC or the host's local timezone -- see
    # require_timezone_aware().
    db = self.make_db()
    deployment_id, _ = self.insert_deployment_with_recordings(db)
    with self.assertRaises(ValueError):
      db.insert_recording(
          filename='dated.wav',
          deployment_id=deployment_id,
          datetime=dt.datetime(2024, 1, 1, 10, 0),  # Naive.
      )

  def test_insert_recording_idempotent_after_it_has_annotations(self):
    # Same DuckDB upsert-vs-FK limitation, but for recordings referenced by
    # annotations/detections.
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[0.0, 1.0],
        label='x',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )

    same_id = db.insert_recording(
        filename='f0.wav', deployment_id=deployment_id
    )
    self.assertEqual(same_id, recording_ids[0])

  def test_get_all_projects(self):
    db = self.make_db()
    db.insert_deployment(name='d1', project='alpha_project')
    db.insert_deployment(name='d2', project='beta_project')
    self.assertEqual(
        list(db.get_all_projects()), ['alpha_project', 'beta_project']
    )

  def test_get_all_deployments_wrong_filter_column_raises_clearly(self):
    db = self.make_db()
    db.insert_deployment(name='d1', project='p1')
    with self.assertRaises(ValueError) as cm:
      db.get_all_deployments(
          filter=config_dict.create(eq=dict(deployment_id=1))
      )
    self.assertIn('deployment_id', str(cm.exception))
    self.assertIn('id', str(cm.exception))

  def test_get_all_recordings_wrong_filter_column_raises_clearly(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.get_all_recordings(filter=config_dict.create(eq=dict(not_a_col=1)))

  def test_get_all_annotations_wrong_filter_column_raises_clearly(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.get_all_annotations(filter=config_dict.create(eq=dict(not_a_col=1)))

  def test_remove_recording_not_implemented(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    with self.assertRaises(NotImplementedError):
      db.remove_recording(recording_ids[0])

  def test_remove_deployment_cascades_and_deletes_parquet(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    scores = np.random.RandomState(0).rand(1, len(CLASS_LABELS)).astype(
        np.float16
    )
    db.insert_deployment_scores(deployment_id, recording_ids, [0.0], [3.0], scores)
    parquet_path = db.deployment_parquet_path(deployment_id)
    self.assertTrue(parquet_path.exists())

    db.remove_deployment(deployment_id)

    self.assertEmpty(db.get_all_deployments())
    self.assertEmpty(db.get_all_recordings())
    self.assertEmpty(db.get_all_annotations())
    self.assertEmpty(db.get_all_detections())
    self.assertFalse(parquet_path.exists())

  def test_remove_deployment_missing_raises_keyerror(self):
    db = self.make_db()
    with self.assertRaises(KeyError):
      db.remove_deployment(12345)

  # ---------------------------------------------------------------------
  # annotations.
  # ---------------------------------------------------------------------

  def test_insert_and_get_annotation(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    annotation_id = db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    annotation = db.get_annotation(annotation_id)
    self.assertEqual(annotation.recording_id, recording_ids[0])
    self.assertEqual(annotation.offsets, [0.0, 3.0])
    self.assertEqual(annotation.label, 'bird')
    self.assertEqual(annotation.label_type, datatypes.LabelType.POSITIVE)

  def test_insert_annotation_duplicate_handling(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    kwargs = dict(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    first_id = db.insert_annotation(**kwargs)

    with self.assertRaises(RuntimeError):
      db.insert_annotation(**kwargs)

    skip_id = db.insert_annotation(handle_duplicates='skip', **kwargs)
    self.assertEqual(skip_id, first_id)

    self.assertLen(db.get_all_annotations(), 1)

  def test_get_all_labels_and_count_each_label(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[3.0, 6.0],
        label='frog',
        label_type=datatypes.LabelType.NEGATIVE,
        provenance='me',
    )
    self.assertEqual(list(db.get_all_labels()), ['bird', 'frog'])
    self.assertEqual(
        list(db.get_all_labels(label_type=datatypes.LabelType.POSITIVE)),
        ['bird'],
    )
    self.assertEqual(db.count_each_label()['bird'], 1)

  def test_remove_annotation(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    annotation_id = db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    db.remove_annotation(annotation_id)
    self.assertEmpty(db.get_all_annotations())
    with self.assertRaises(KeyError):
      db.remove_annotation(annotation_id)

  # ---------------------------------------------------------------------
  # detections.
  # ---------------------------------------------------------------------

  def test_insert_and_get_detection(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    detection_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    detection = db.get_detection(detection_id)
    self.assertEqual(detection.recording_id, recording_ids[0])
    self.assertEqual(detection.offsets, [0.0, 3.0])
    self.assertEqual(detection.class_name, 'alpha')
    self.assertEqual(detection.score, 0.9)

  def test_insert_detection_duplicate_overwrite_updates_score(self):
    # Regression test: re-inserting a detection with the same
    # (recording_id, offsets, class_name) used to always create a second,
    # duplicate row -- e.g. re-running add_detections_from_threshold with
    # the same settings. The default 'overwrite' mode now updates the
    # existing row's score in place instead.
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    first_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.5,
    )
    second_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    self.assertEqual(second_id, first_id)
    self.assertEqual(db.get_detection(first_id).score, 0.9)
    self.assertLen(db.get_all_detections(), 1)

  def test_insert_detection_duplicate_skip_keeps_existing(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    first_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.5,
    )
    second_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
        handle_duplicates='skip',
    )
    self.assertEqual(second_id, first_id)
    self.assertEqual(db.get_detection(first_id).score, 0.5)
    self.assertLen(db.get_all_detections(), 1)

  def test_insert_detection_duplicate_error_raises(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.5,
    )
    with self.assertRaises(RuntimeError):
      db.insert_detection(
          recording_id=recording_ids[0],
          offsets=[0.0, 3.0],
          class_name='alpha',
          score=0.9,
          handle_duplicates='error',
      )
    self.assertLen(db.get_all_detections(), 1)

  def test_insert_detection_distinct_identity_does_not_conflict(self):
    # Different offsets, or a different class_name, are distinct
    # detections even for the same recording -- only an exact
    # (recording_id, offsets, class_name) match is a duplicate.
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    id1 = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.5,
    )
    id2 = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[3.0, 6.0],
        class_name='alpha',
        score=0.5,
    )
    id3 = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='beta',
        score=0.5,
    )
    self.assertLen({id1, id2, id3}, 3)
    self.assertLen(db.get_all_detections(), 3)

  def test_insert_detections_batch_handle_duplicates_applies_to_all(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.5,
    )
    [updated_id] = db.insert_detections_batch(
        [
            dict(
                recording_id=recording_ids[0],
                offsets=[0.0, 3.0],
                class_name='alpha',
                score=0.9,
            )
        ],
        handle_duplicates='overwrite',
    )
    self.assertEqual(db.get_detection(updated_id).score, 0.9)
    self.assertLen(db.get_all_detections(), 1)

  def test_insert_detections_batch(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    detections_batch = [
        dict(
            recording_id=recording_ids[0],
            offsets=[i * 3.0, i * 3.0 + 3.0],
            class_name='alpha',
            score=0.5 + i * 0.1,
        )
        for i in range(3)
    ]
    detection_ids = db.insert_detections_batch(detections_batch)
    self.assertLen(detection_ids, 3)
    self.assertLen(db.get_all_detections(), 3)

  def test_get_detection_annotations(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    detection_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[1.0, 2.0],  # intersects the detection.
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    db.insert_annotation(
        recording_id=recording_ids[0],
        offsets=[10.0, 12.0],  # does not intersect.
        label='frog',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='me',
    )
    annotations = db.get_detection_annotations(detection_id)
    self.assertLen(annotations, 1)
    self.assertEqual(annotations[0].label, 'bird')

  def test_remove_detection(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    detection_id = db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    db.remove_detection(detection_id)
    self.assertEmpty(db.get_all_detections())
    with self.assertRaises(KeyError):
      db.remove_detection(detection_id)

  def test_get_all_detections_with_filters(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, num_recordings=2
    )
    db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.9,
    )
    db.insert_detection(
        recording_id=recording_ids[1],
        offsets=[0.0, 3.0],
        class_name='beta',
        score=0.8,
    )
    matches = db.get_all_detections(
        recordings_filter=config_dict.create(eq=dict(filename='f0.wav'))
    )
    self.assertLen(matches, 1)
    self.assertEqual(matches[0].class_name, 'alpha')

    matches = db.get_all_detections(
        filter=config_dict.create(eq=dict(class_name='beta'))
    )
    self.assertLen(matches, 1)
    self.assertEqual(matches[0].recording_id, recording_ids[1])

  def test_get_all_detections_wrong_filter_column_raises_clearly(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.get_all_detections(filter=config_dict.create(eq=dict(not_a_col=1)))
    with self.assertRaises(ValueError):
      db.get_all_detections(
          deployments_filter=config_dict.create(eq=dict(deployment_id=1))
      )
    with self.assertRaises(ValueError):
      db.get_all_detections(
          recordings_filter=config_dict.create(eq=dict(not_a_col=1))
      )

  # ---------------------------------------------------------------------
  # select_detections.
  # ---------------------------------------------------------------------

  def make_multi_deployment_detections(
      self, num_recordings=10, num_deployments=2, seed=0
  ):
    """Sets up `num_deployments` deployments with detections on each.

    Mirrors `make_multi_deployment_scores()`, but inserts individual
    `detections` rows (alternating class between 'alpha' and 'beta') rather
    than a full class-score Parquet matrix.
    """
    db = self.make_db()
    rng = np.random.RandomState(seed)
    deployment_ids = []
    for d in range(num_deployments):
      base_date = dt.datetime(
          2024, 1, 1, tzinfo=dt.timezone.utc
      ) + dt.timedelta(days=31 * d)
      deployment_id, recording_ids = self.insert_deployment_with_recordings(
          db, num_recordings=num_recordings, name=f'd{d}'
      )
      for i, recording_id in enumerate(recording_ids):
        db.db.execute(
            'UPDATE recordings SET datetime = ? WHERE id = ?',
            [base_date + dt.timedelta(days=i), recording_id],
        )
        db.insert_detection(
            recording_id=recording_id,
            offsets=[0.0, 3.0],
            class_name='alpha' if i % 2 == 0 else 'beta',
            score=float(rng.rand()),
        )
      deployment_ids.append(deployment_id)
    return db, deployment_ids

  def test_select_detections_requires_n(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_detections(selection='top', n=None)

  def test_select_detections_binned_requires_bin_edges(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_detections(selection='binned', n=1)

  def test_select_detections_unknown_class_raises(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_detections(class_name='not_a_real_class', selection='top', n=1)

  def test_select_detections_empty_database_returns_empty(self):
    db = self.make_db()
    result = db.select_detections(selection='top', n=5)
    self.assertEmpty(result)
    self.assertIn('score', result.columns)
    self.assertNotIn('stratum', result.columns)

  def test_select_detections_top_across_all_classes(self):
    db, _ = self.make_multi_deployment_detections()
    result = db.select_detections(selection='top', n=3)
    self.assertLen(result, 3)
    self.assertEqual(
        list(result['score']), sorted(result['score'], reverse=True)
    )

  def test_select_detections_class_name_filter(self):
    db, _ = self.make_multi_deployment_detections()
    result = db.select_detections(class_name='alpha', selection='top', n=100)
    self.assertNotEmpty(result)
    self.assertTrue((result['class_name'] == 'alpha').all())

  def test_select_detections_includes_recording_and_deployment_metadata(self):
    db, deployment_ids = self.make_multi_deployment_detections(
        num_deployments=1
    )
    result = db.select_detections(selection='top', n=1)
    self.assertLen(result, 1)
    self.assertEqual(result['deployment_id'].iloc[0], deployment_ids[0])
    self.assertEqual(result['deployment_name'].iloc[0], 'd0')
    self.assertRegex(result['recording_filename'].iloc[0], r'^f\d+\.wav$')
    self.assertIsNotNone(result['recording_datetime'].iloc[0])
    self.assertIn('detection_id', result.columns)

  def test_select_detections_top_stratified_by_deployment(self):
    db, deployment_ids = self.make_multi_deployment_detections(
        num_deployments=2
    )
    result = db.select_detections(
        selection='top', n=2, by_deployment=True
    )
    self.assertLen(result, 4)
    self.assertEqual(set(result['stratum']), set(deployment_ids))
    for deployment_id in deployment_ids:
      sub = result[result['stratum'] == deployment_id]
      self.assertEqual(
          list(sub['score']), sorted(sub['score'], reverse=True)
      )

  def test_select_detections_deployments_filter(self):
    db, deployment_ids = self.make_multi_deployment_detections()
    result = db.select_detections(
        selection='top',
        n=100,
        deployments_filter=config_dict.create(eq=dict(name='d0')),
    )
    self.assertEqual(set(result['deployment_id']), {deployment_ids[0]})

  def test_select_detections_deployments_filter_wrong_column_raises(self):
    db, _ = self.make_multi_deployment_detections()
    with self.assertRaises(ValueError):
      db.select_detections(
          selection='top',
          n=1,
          deployments_filter=config_dict.create(eq=dict(deployment_id=1)),
      )

  def test_select_detections_filter_wrong_column_raises(self):
    db, _ = self.make_multi_deployment_detections()
    with self.assertRaises(ValueError):
      db.select_detections(
          selection='top',
          n=1,
          filter=config_dict.create(eq=dict(not_a_column=1)),
      )

  def test_select_detections_extra_column_filter(self):
    db, _ = self.make_multi_deployment_detections()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, name='d_manual'
    )
    db.insert_detection(
        recording_id=recording_ids[0],
        offsets=[0.0, 3.0],
        class_name='alpha',
        score=0.99,
        provenance='manual',
    )
    result = db.select_detections(
        selection='top',
        n=100,
        filter=config_dict.create(eq=dict(provenance='manual')),
    )
    self.assertLen(result, 1)
    self.assertAlmostEqual(result['score'].iloc[0], 0.99, places=4)

  def test_select_detections_min_max_score_filter(self):
    db, _ = self.make_multi_deployment_detections()
    result = db.select_detections(
        selection='top', n=1000, min_score=0.3, max_score=0.7
    )
    self.assertGreaterEqual(result['score'].min(), 0.3)
    self.assertLessEqual(result['score'].max(), 0.7)

  def test_select_detections_start_end_date_filter(self):
    db, deployment_ids = self.make_multi_deployment_detections()
    result = db.select_detections(
        selection='top',
        n=1000,
        start_date=dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc),
        end_date=dt.datetime(2024, 2, 5, tzinfo=dt.timezone.utc),
    )
    self.assertNotEmpty(result)
    self.assertEqual(set(result['deployment_id']), {deployment_ids[1]})

  def test_select_detections_naive_start_end_date_raises(self):
    db, _ = self.make_multi_deployment_detections()
    with self.assertRaises(ValueError):
      db.select_detections(
          selection='top', n=10, start_date=dt.datetime(2024, 2, 1)
      )

  def test_select_detections_random_seed_is_reproducible(self):
    db, _ = self.make_multi_deployment_detections(num_deployments=3)
    first = db.select_detections(selection='random', n=5, random_seed=42)
    second = db.select_detections(selection='random', n=5, random_seed=42)
    self.assertEqual(
        first['detection_id'].tolist(), second['detection_id'].tolist()
    )

  def test_select_detections_random_different_seeds_differ(self):
    db, _ = self.make_multi_deployment_detections(num_deployments=3)
    first = db.select_detections(selection='random', n=5, random_seed=1)
    second = db.select_detections(selection='random', n=5, random_seed=2)
    self.assertNotEqual(
        first['detection_id'].tolist(), second['detection_id'].tolist()
    )

  def test_select_detections_binned_percentile_bins(self):
    db, _ = self.make_multi_deployment_detections(
        num_recordings=40, num_deployments=1
    )
    result = db.select_detections(
        selection='binned', n=2, bin_edges=[0, 0.5, 0.75, 0.875, 1]
    )
    self.assertContainsSubset(set(result['bin_idx']), {0, 1, 2, 3})
    bin_maxes = result.groupby('bin_idx')['score'].max()
    bin_mins = result.groupby('bin_idx')['score'].min()
    for bin_idx in sorted(bin_maxes.index)[:-1]:
      self.assertLessEqual(bin_maxes[bin_idx], bin_mins[bin_idx + 1] + 1e-6)

  def test_select_detections_by_date_week(self):
    db, _ = self.make_multi_deployment_detections(
        num_recordings=14, num_deployments=1
    )
    result = db.select_detections(selection='top', n=1, date_bins='week')
    self.assertGreaterEqual(len(result), 2)
    self.assertIn('date_bin_label', result.columns)

  def test_select_detections_by_custom_date_bins(self):
    db, _ = self.make_multi_deployment_detections()
    custom_bins = [
        (dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), 'first'),
        (dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc), 'second'),
    ]
    result = db.select_detections(
        selection='top', n=2, date_bins=custom_bins
    )
    self.assertEqual(set(result['stratum']), {'first', 'second'})
    self.assertEqual(set(result['date_bin_label']), {'first', 'second'})

  def test_select_detections_by_deployment_and_date(self):
    db, deployment_ids = self.make_multi_deployment_detections(
        num_deployments=2
    )
    result = db.select_detections(
        selection='top', n=1, by_deployment=True, date_bins='week'
    )
    self.assertIn('date_bin_label', result.columns)
    for stratum in result['stratum']:
      self.assertIn('deployment_id', stratum)
      self.assertIn('date_bin', stratum)
      self.assertIn(stratum['deployment_id'], deployment_ids)

  # ---------------------------------------------------------------------
  # Score storage / retrieval.
  # ---------------------------------------------------------------------

  def test_get_embedding_dim_and_dtype(self):
    db = self.make_db()
    self.assertEqual(db.get_embedding_dim(), len(CLASS_LABELS))
    self.assertEqual(db.get_embedding_dtype(), np.float16)

  def test_insert_deployment_scores_validates_shape(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    bad_scores = np.zeros((1, len(CLASS_LABELS) + 1), dtype=np.float16)
    with self.assertRaises(ValueError):
      db.insert_deployment_scores(
          deployment_id, recording_ids, [0.0], [3.0], bad_scores
      )

  def test_insert_deployment_scores_validates_lengths(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    scores = np.zeros((1, len(CLASS_LABELS)), dtype=np.float16)
    with self.assertRaises(ValueError):
      db.insert_deployment_scores(
          deployment_id, recording_ids, [0.0, 1.0], [3.0], scores
      )

  def test_insert_deployment_scores_errors_on_existing_file_by_default(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    scores = np.zeros((1, len(CLASS_LABELS)), dtype=np.float16)
    db.insert_deployment_scores(deployment_id, recording_ids, [0.0], [3.0], scores)
    with self.assertRaises(ValueError):
      db.insert_deployment_scores(
          deployment_id, recording_ids, [0.0], [3.0], scores
      )

  def test_insert_deployment_scores_overwrite(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    scores_1 = np.zeros((1, len(CLASS_LABELS)), dtype=np.float16)
    scores_2 = np.ones((1, len(CLASS_LABELS)), dtype=np.float16)
    db.insert_deployment_scores(
        deployment_id, recording_ids, [0.0], [3.0], scores_1
    )
    db.insert_deployment_scores(
        deployment_id,
        recording_ids,
        [0.0],
        [3.0],
        scores_2,
        handle_existing='overwrite',
    )
    batch = db.get_scores_batch(deployment_id, recording_ids, [0.0], [3.0])
    np.testing.assert_allclose(batch, scores_2)

  def test_get_scores_batch_round_trip(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    n = 5
    recording_ids = recording_ids * n
    start_times = [float(i * 3) for i in range(n)]
    end_times = [float(i * 3 + 3) for i in range(n)]
    scores = np.random.RandomState(0).rand(n, len(CLASS_LABELS)).astype(
        np.float16
    )
    db.insert_deployment_scores(
        deployment_id, recording_ids, start_times, end_times, scores
    )

    batch = db.get_scores_batch(
        deployment_id, recording_ids, start_times, end_times
    )
    np.testing.assert_allclose(batch, scores, atol=1e-5)

  def test_get_scores_batch_missing_key_raises_keyerror(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    scores = np.zeros((1, len(CLASS_LABELS)), dtype=np.float16)
    db.insert_deployment_scores(deployment_id, recording_ids, [0.0], [3.0], scores)
    with self.assertRaises(KeyError):
      db.get_scores_batch(deployment_id, recording_ids, [999.0], [1002.0])

  def make_multi_deployment_scores(
      self, num_recordings=10, num_deployments=2, seed=0
  ):
    """Sets up `num_deployments` deployments, each with scored recordings.

    Recording `i` of deployment `d` gets `datetime` `base_date(d) +
    i days`, where `base_date` differs per deployment (so deployments are
    also separable by date range), and random `alpha`/other-class scores.
    """
    db = self.make_db()
    rng = np.random.RandomState(seed)
    deployment_ids = []
    recording_ids_by_deployment = {}
    for d in range(num_deployments):
      base_date = dt.datetime(
          2024, 1, 1, tzinfo=dt.timezone.utc
      ) + dt.timedelta(days=31 * d)
      deployment_id, recording_ids = self.insert_deployment_with_recordings(
          db,
          num_recordings=num_recordings,
          name=f'd{d}',
      )
      for i, recording_id in enumerate(recording_ids):
        db.db.execute(
            'UPDATE recordings SET datetime = ? WHERE id = ?',
            [base_date + dt.timedelta(days=i), recording_id],
        )
      scores = rng.rand(num_recordings, len(CLASS_LABELS)).astype(np.float16)
      db.insert_deployment_scores(
          deployment_id,
          recording_ids,
          [0.0] * num_recordings,
          [3.0] * num_recordings,
          scores,
      )
      deployment_ids.append(deployment_id)
      recording_ids_by_deployment[deployment_id] = recording_ids
    return db, deployment_ids, recording_ids_by_deployment

  @parameterized.parameters('American Redstart', 'a-b', 'alpha')
  def test_select_scores_handles_various_class_names(self, class_name):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(db)
    scores = np.zeros((1, len(CLASS_LABELS)), dtype=np.float16)
    class_idx = CLASS_LABELS.index(class_name)
    scores[0, class_idx] = 0.99
    db.insert_deployment_scores(deployment_id, recording_ids, [0.0], [3.0], scores)

    result = db.select_scores(class_name, selection='top', n=1, min_score=0.5)
    self.assertLen(result, 1)
    # float16 has ~3 decimal digits of precision, so 0.99 round-trips as
    # ~0.9902 -- assert to float16 precision, not float32/float64 precision.
    self.assertAlmostEqual(result['score'].iloc[0], 0.99, places=2)

  def test_select_scores_unknown_class_raises(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_scores('not_a_real_class', selection='top', n=1)

  def test_select_scores_requires_n(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_scores('alpha', selection='top', n=None)

  def test_select_scores_binned_requires_bin_edges(self):
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.select_scores('alpha', selection='binned', n=1)

  def test_select_scores_top_across_multiple_deployments(self):
    db, _, _ = self.make_multi_deployment_scores()
    result = db.select_scores('alpha', selection='top', n=3)
    self.assertLen(result, 3)
    self.assertEqual(
        list(result['score']), sorted(result['score'], reverse=True)
    )
    self.assertNotIn('stratum', result.columns)
    self.assertNotIn('date_bin_label', result.columns)

  def test_select_scores_includes_recording_and_deployment_metadata(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores(
        num_deployments=1
    )
    result = db.select_scores('alpha', selection='top', n=1)
    self.assertLen(result, 1)
    self.assertEqual(result['deployment_id'].iloc[0], deployment_ids[0])
    self.assertEqual(result['deployment_name'].iloc[0], 'd0')
    self.assertRegex(result['recording_filename'].iloc[0], r'^f\d+\.wav$')
    self.assertIsNotNone(result['recording_datetime'].iloc[0])

  def test_select_scores_top_stratified_by_deployment(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores(
        num_deployments=2
    )
    result = db.select_scores(
        'alpha', selection='top', n=2, by_deployment=True
    )
    self.assertLen(result, 4)  # 2 per deployment.
    self.assertEqual(set(result['stratum']), set(deployment_ids))
    self.assertNotIn('date_bin_label', result.columns)
    for deployment_id in deployment_ids:
      sub = result[result['stratum'] == deployment_id]
      self.assertEqual(
          list(sub['score']), sorted(sub['score'], reverse=True)
      )

  def test_select_scores_deployments_filter(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores()
    result = db.select_scores(
        'alpha',
        selection='top',
        n=100,
        deployments_filter=config_dict.create(eq=dict(name='d0')),
    )
    self.assertEqual(set(result['deployment_id']), {deployment_ids[0]})

  def test_select_scores_deployments_filter_wrong_column_raises_clearly(self):
    # Regression test: `deployments_filter` must use the deployments
    # table's own primary key column name, `id` -- NOT `deployment_id`
    # (which only exists as a foreign key on recordings/detections, and is
    # also the name of a column in select_scores()'s own OUTPUT frame,
    # making this an easy mistake). This must raise a clear ValueError
    # naming the actual valid columns, not a raw duckdb.BinderException.
    db, _, _ = self.make_multi_deployment_scores()
    with self.assertRaises(ValueError) as cm:
      db.select_scores(
          'alpha',
          selection='top',
          n=1,
          deployments_filter=config_dict.create(eq=dict(deployment_id=1)),
      )
    self.assertIn('deployment_id', str(cm.exception))
    self.assertIn('id', str(cm.exception))

  def test_select_scores_recordings_filter_wrong_column_raises_clearly(self):
    db, _, _ = self.make_multi_deployment_scores()
    with self.assertRaises(ValueError):
      db.select_scores(
          'alpha',
          selection='top',
          n=1,
          recordings_filter=config_dict.create(eq=dict(not_a_column=1)),
      )

  def test_select_scores_min_max_score_filter(self):
    db, _, _ = self.make_multi_deployment_scores()
    result = db.select_scores(
        'alpha', selection='top', n=1000, min_score=0.3, max_score=0.7
    )
    self.assertGreaterEqual(result['score'].min(), 0.3)
    self.assertLessEqual(result['score'].max(), 0.7)

  def test_select_scores_start_end_date_filter(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores()
    # Deployment 0's recordings start 2024-01-01; deployment 1's start
    # 2024-02-01 (see make_multi_deployment_scores). A date range entirely
    # within deployment 1's span should only return deployment 1's rows.
    result = db.select_scores(
        'alpha',
        selection='top',
        n=1000,
        start_date=dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc),
        end_date=dt.datetime(2024, 2, 5, tzinfo=dt.timezone.utc),
    )
    self.assertNotEmpty(result)
    self.assertEqual(set(result['deployment_id']), {deployment_ids[1]})

  def test_select_scores_naive_start_end_date_raises(self):
    db, _, _ = self.make_multi_deployment_scores()
    with self.assertRaises(ValueError):
      db.select_scores(
          'alpha', selection='top', n=10, start_date=dt.datetime(2024, 2, 1)
      )
    with self.assertRaises(ValueError):
      db.select_scores(
          'alpha', selection='top', n=10, end_date=dt.datetime(2024, 2, 5)
      )

  def test_select_scores_random_seed_is_reproducible(self):
    db, _, _ = self.make_multi_deployment_scores(num_deployments=3)
    first = db.select_scores(
        'alpha', selection='random', n=5, random_seed=42
    )
    second = db.select_scores(
        'alpha', selection='random', n=5, random_seed=42
    )
    self.assertEqual(
        first['recording_id'].tolist(), second['recording_id'].tolist()
    )

  def test_select_scores_random_different_seeds_differ(self):
    db, _, _ = self.make_multi_deployment_scores(num_deployments=3)
    first = db.select_scores(
        'alpha', selection='random', n=5, random_seed=1
    )
    second = db.select_scores(
        'alpha', selection='random', n=5, random_seed=2
    )
    self.assertNotEqual(
        first['recording_id'].tolist(), second['recording_id'].tolist()
    )

  def test_select_scores_binned_percentile_bins(self):
    db, _, _ = self.make_multi_deployment_scores(
        num_recordings=40, num_deployments=1
    )
    result = db.select_scores(
        'alpha',
        selection='binned',
        n=2,
        bin_edges=[0, 0.5, 0.75, 0.875, 1],
    )
    self.assertContainsSubset(set(result['bin_idx']), {0, 1, 2, 3})
    # Higher bins should have higher (or equal) scores than lower bins.
    bin_maxes = result.groupby('bin_idx')['score'].max()
    bin_mins = result.groupby('bin_idx')['score'].min()
    for bin_idx in sorted(bin_maxes.index)[:-1]:
      self.assertLessEqual(bin_maxes[bin_idx], bin_mins[bin_idx + 1] + 1e-6)

  def test_select_scores_binned_stratified_by_deployment(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores(
        num_recordings=20, num_deployments=2
    )
    result = db.select_scores(
        'alpha',
        selection='binned',
        n=1,
        bin_edges=[0, 0.5, 1],
        by_deployment=True,
    )
    # Up to 2 bins * 1 per bin * 2 deployments.
    self.assertLessEqual(len(result), 4)
    self.assertContainsSubset(set(result['stratum']), set(deployment_ids))

  def test_select_scores_by_date_week(self):
    db, _, _ = self.make_multi_deployment_scores(
        num_recordings=14, num_deployments=1
    )
    result = db.select_scores(
        'alpha',
        selection='top',
        n=1,
        date_bins='week',
    )
    # 14 daily recordings starting 2024-01-01 should span 3 week-buckets.
    self.assertGreaterEqual(len(result), 2)
    self.assertIn('date_bin_label', result.columns)
    self.assertNotEmpty(result['date_bin_label'].dropna())

  def test_select_scores_by_custom_date_bins(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores()
    custom_bins = [
        (dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc), 'first'),
        (dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc), 'second'),
    ]
    result = db.select_scores(
        'alpha',
        selection='top',
        n=2,
        date_bins=custom_bins,
    )
    self.assertEqual(set(result['stratum']), {'first', 'second'})
    self.assertEqual(set(result['date_bin_label']), {'first', 'second'})

  def test_select_scores_naive_custom_date_bins_raises(self):
    db, _, _ = self.make_multi_deployment_scores()
    with self.assertRaises(ValueError):
      db.select_scores(
          'alpha',
          selection='top',
          n=2,
          date_bins=[(dt.datetime(2024, 1, 1), 'first')],
      )

  def test_select_scores_by_deployment_and_date(self):
    db, deployment_ids, _ = self.make_multi_deployment_scores(
        num_deployments=2
    )
    result = db.select_scores(
        'alpha',
        selection='top',
        n=1,
        by_deployment=True,
        date_bins='week',
    )
    self.assertIn('date_bin_label', result.columns)
    for stratum in result['stratum']:
      self.assertIn('deployment_id', stratum)
      self.assertIn('date_bin', stratum)
      self.assertIn(stratum['deployment_id'], deployment_ids)

  def test_select_scores_no_matching_deployments_returns_empty(self):
    db, _, _ = self.make_multi_deployment_scores()
    result = db.select_scores(
        'alpha',
        selection='top',
        n=5,
        deployments_filter=config_dict.create(eq=dict(name='nonexistent')),
    )
    self.assertEmpty(result)
    self.assertIn('score', result.columns)

  def test_select_scores_no_scores_written_returns_empty(self):
    db = self.make_db()
    self.insert_deployment_with_recordings(db)  # No insert_deployment_scores.
    result = db.select_scores('alpha', selection='top', n=5)
    self.assertEmpty(result)

  def test_add_detections_from_threshold(self):
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, num_recordings=3
    )
    scores = np.zeros((3, len(CLASS_LABELS)), dtype=np.float16)
    scores[0, 0] = 0.95  # above threshold
    scores[1, 0] = 0.1  # below threshold
    scores[2, 0] = 0.99  # above threshold
    start_times = [0.0, 0.0, 0.0]
    end_times = [3.0, 3.0, 3.0]
    db.insert_deployment_scores(
        deployment_id, recording_ids, start_times, end_times, scores
    )

    detection_ids = db.add_detections_from_threshold(
        threshold=0.5, class_name='alpha', provenance='auto'
    )
    self.assertLen(detection_ids, 2)
    detections = db.get_all_detections()
    self.assertLen(detections, 2)
    for detection in detections:
      self.assertEqual(detection.class_name, 'alpha')
      self.assertGreaterEqual(detection.score, 0.5)
      self.assertEqual(detection.provenance, 'auto')

  def test_add_detections_from_threshold_unknown_class_raises_even_when_empty(
      self,
  ):
    # Regression test: class-name validation must happen even when there
    # are no deployments/scores yet to scan -- it must not be skipped just
    # because the Parquet-scanning code path is never reached.
    db = self.make_db()
    with self.assertRaises(ValueError):
      db.add_detections_from_threshold(
          threshold=0.5, class_name='not_a_real_class'
      )

  def test_add_detections_from_threshold_no_deployments_returns_empty(self):
    db = self.make_db()
    self.assertEmpty(
        db.add_detections_from_threshold(threshold=0.5, class_name='alpha')
    )

  def test_add_detections_from_threshold_rerun_is_idempotent(self):
    # Regression test for the originally-reported bug: re-running
    # add_detections_from_threshold with the same settings must not
    # duplicate detections (default handle_duplicates='overwrite').
    db = self.make_db()
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, num_recordings=3
    )
    scores = np.zeros((3, len(CLASS_LABELS)), dtype=np.float16)
    scores[0, 0] = 0.95
    scores[2, 0] = 0.99
    db.insert_deployment_scores(
        deployment_id, recording_ids, [0.0] * 3, [3.0] * 3, scores
    )

    first_ids = db.add_detections_from_threshold(
        threshold=0.5, class_name='alpha'
    )
    second_ids = db.add_detections_from_threshold(
        threshold=0.5, class_name='alpha'
    )
    self.assertLen(db.get_all_detections(), 2)
    self.assertEqual(set(first_ids), set(second_ids))

  def test_add_detections_from_threshold_all_classes(self):
    # Omitting class_name scans every class in one query (via UNPIVOT), so
    # a single window can yield multiple detections, one per qualifying
    # class.
    db = self.make_db(class_labels=('alpha', 'beta'))
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, num_recordings=2
    )
    scores = np.array([[0.95, 0.1], [0.05, 0.99]], dtype=np.float16)
    db.insert_deployment_scores(
        deployment_id, recording_ids, [0.0, 0.0], [3.0, 3.0], scores
    )

    detection_ids = db.add_detections_from_threshold(threshold=0.9)
    detections = db.get_all_detections()
    self.assertLen(detection_ids, 2)
    self.assertEqual(
        {d.class_name for d in detections}, {'alpha', 'beta'}
    )
    for d in detections:
      self.assertGreaterEqual(d.score, 0.9)

  def test_add_detections_from_threshold_all_classes_handle_duplicates(self):
    db = self.make_db(class_labels=('alpha', 'beta'))
    deployment_id, recording_ids = self.insert_deployment_with_recordings(
        db, num_recordings=1
    )
    scores = np.array([[0.95, 0.1]], dtype=np.float16)
    db.insert_deployment_scores(
        deployment_id, recording_ids, [0.0], [3.0], scores
    )
    db.add_detections_from_threshold(threshold=0.9)

    # Re-scan with an updated score for the same window/class: overwrite
    # (default) should update in place, not duplicate.
    db.insert_deployment_scores(
        deployment_id,
        recording_ids,
        [0.0],
        [3.0],
        np.array([[0.5, 0.1]], dtype=np.float16),
        handle_existing='overwrite',
    )
    # Nothing scores >= 0.9 anymore, so a re-scan finds no new detections,
    # but the previously-inserted one should still be untouched.
    db.add_detections_from_threshold(threshold=0.9)
    self.assertLen(db.get_all_detections(), 1)

  # ---------------------------------------------------------------------
  # Unsupported interface methods.
  # ---------------------------------------------------------------------

  def test_window_and_embedding_methods_not_implemented(self):
    db = self.make_db()
    _, recording_ids = self.insert_deployment_with_recordings(db)
    with self.assertRaises(NotImplementedError):
      db.insert_window(recording_id=recording_ids[0], offsets=[0.0, 1.0])
    with self.assertRaises(NotImplementedError):
      db.get_window(window_id=1)
    with self.assertRaises(NotImplementedError):
      db.remove_window(window_id=1)
    with self.assertRaises(NotImplementedError):
      db.get_window_annotations(window_id=1)
    with self.assertRaises(NotImplementedError):
      db.get_embedding(window_id=1)
    with self.assertRaises(NotImplementedError):
      db.count_embeddings()
    with self.assertRaises(NotImplementedError):
      db.match_window_ids()
    with self.assertRaises(NotImplementedError):
      db.get_all_windows()
    with self.assertRaises(NotImplementedError):
      db.search(query_embedding=np.zeros(len(CLASS_LABELS)), search_list_size=5)


if __name__ == '__main__':
  absltest.main()
