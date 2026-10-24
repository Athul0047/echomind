"""Tests for evaluation row and report generation, without model/audio loading."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from echomind.evaluation import (
    RECORDINGS,
    ChunkEvaluation,
    EvaluationLabel,
    RecordingEvaluation,
    evaluation_rows,
    write_evaluation_csv,
)


class EvaluationWorkflowTests(unittest.TestCase):
    def test_recording_conditions_match_the_three_named_inputs(self) -> None:
        self.assertEqual(
            [(item.filename, item.condition) for item in RECORDINGS],
            [
                ("Test1 - 23:10:26, 8.25 PM.wav", "speech with ceiling fan"),
                ("fan_only.wav", "fan only"),
                ("speech_only.wav", "speech only"),
            ],
        )

    def test_rows_include_overall_and_each_chunk_label_and_score(self) -> None:
        result = RecordingEvaluation(
            filename="sample.wav",
            condition="speech only",
            duration_seconds=2.5,
            chunks=(ChunkEvaluation(
                0.0,
                2.5,
                (EvaluationLabel("Speech", 0.75),),
            ),),
            overall_labels=(EvaluationLabel("Speech", 0.70),),
        )
        rows = evaluation_rows([result])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["chunk"], "overall")
        self.assertEqual(rows[1]["chunk"], "chunk 1 [00:00.00-00:02.50]")
        for row in rows:
            self.assertEqual(row["recording"], "sample.wav")
            self.assertEqual(row["condition"], "speech only")
            self.assertEqual(row["label"], "Speech")
            self.assertIn("score", row)

    def test_csv_is_written_and_existing_report_is_not_overwritten(self) -> None:
        result = RecordingEvaluation(
            filename="sample.wav",
            condition="fan only",
            duration_seconds=1.0,
            chunks=(),
            overall_labels=(EvaluationLabel("Fan", 0.8),),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = write_evaluation_csv([result], temporary_directory)
            first_contents = first.read_bytes()
            second = write_evaluation_csv([result], temporary_directory)

            self.assertEqual(first.name, "three_recording_evaluation.csv")
            self.assertEqual(second.name, "three_recording_evaluation_2.csv")
            self.assertEqual(first.read_bytes(), first_contents)
            with second.open(newline="", encoding="utf-8") as report_file:
                rows = list(csv.DictReader(report_file))
            self.assertEqual(rows[0]["condition"], "fan only")
            self.assertEqual(rows[0]["score"], "0.800000")


if __name__ == "__main__":
    unittest.main()
