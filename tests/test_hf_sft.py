"""Behavioral checks for the rendered-token to HF trainer interface."""

import unittest

from smithtune.hf_rendering import TokenDatum
from smithtune.providers.base import PipelineError
from smithtune.training.hf_sft import to_trainer_example


class HFTrainerInputTests(unittest.TestCase):
    def test_preserves_context_and_supervises_only_the_selected_target(self):
        datum = TokenDatum([10, 20, 30, 40, 50], [0, 0, 0, 1, 1])
        result = to_trainer_example(datum, max_seq_len=5)
        self.assertEqual(result["input_ids"], [10, 20, 30, 40, 50])
        self.assertEqual(result["attention_mask"], [1, 1, 1, 1, 1])
        # Labels are unshifted, including the target's final termination token.
        self.assertEqual(result["labels"], [-100, -100, -100, 40, 50])
        result["input_ids"][0] = 99
        self.assertEqual(datum.token_ids, [10, 20, 30, 40, 50])
        self.assertEqual(datum.token_weights, [0, 0, 0, 1, 1])

    def test_rejects_overlength_targets_instead_of_truncating(self):
        with self.assertRaisesRegex(PipelineError, "reject its whole trajectory"):
            to_trainer_example(TokenDatum([1, 2, 3], [0, 1, 1]), max_seq_len=2)

    def test_rejects_targets_with_no_predictable_supervision(self):
        for weights in ([0, 0], [1, 0]):
            with self.subTest(weights=weights), self.assertRaisesRegex(PipelineError, "causal shifting"):
                to_trainer_example(TokenDatum([1, 2], weights), max_seq_len=2)

    def test_rejects_fractional_or_nonfinite_loss_weights(self):
        for weight in (0.5, float("nan"), float("inf"), -1):
            with self.subTest(weight=weight), self.assertRaisesRegex(PipelineError, "binary"):
                to_trainer_example(TokenDatum([1, 2], [0, weight]), max_seq_len=2)

    def test_rejects_mismatched_or_empty_sequences(self):
        for datum in (TokenDatum([], []), TokenDatum([1, 2], [1])):
            with self.subTest(datum=datum), self.assertRaisesRegex(PipelineError, "equal nonzero length"):
                to_trainer_example(datum, max_seq_len=2)

    def test_rejects_invalid_token_ids(self):
        for token in (-1, True, 1.5):
            with self.subTest(token=token), self.assertRaisesRegex(PipelineError, "token IDs"):
                to_trainer_example(TokenDatum([1, token], [0, 1]), max_seq_len=2)

    def test_rejects_invalid_context_limits(self):
        for limit in (0, -1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(PipelineError, "positive integer"):
                to_trainer_example(TokenDatum([1, 2], [0, 1]), max_seq_len=limit)
