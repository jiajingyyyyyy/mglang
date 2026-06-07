import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.managers.io_struct import GenerateReqInput, StructuredRequestHints


class TestStructuredRequestHints(unittest.TestCase):
    def test_chat_request_accepts_top_level_hints(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            sglang_hints={
                "prefix_key": "prefix-a",
                "decode_class": "open_world",
                "deadline_ms": 25,
                "cache_pin_mode": "layered_static",
                "ignored": "value",
            },
        )

        hints = StructuredRequestHints.from_raw(
            request.sglang_hints, request.metadata
        )

        self.assertEqual(hints.prefix_key, "prefix-a")
        self.assertEqual(hints.decode_class, "open_world")
        self.assertEqual(hints.deadline_ms, 25.0)
        self.assertEqual(hints.cache_pin_mode, "layered_static")
        self.assertFalse(hasattr(hints, "ignored"))

    def test_completion_request_accepts_metadata_fallback(self):
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            metadata={"sglang_hints": {"prefix_key": "prefix-b"}},
        )

        hints = StructuredRequestHints.from_raw(
            request.sglang_hints, request.metadata
        )

        self.assertEqual(hints.prefix_key, "prefix-b")

    def test_invalid_hint_values_are_dropped(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            sglang_hints={
                "prefix_key": 123,
                "deadline_ms": "soon",
                "cache_affinity_key": "worker-group-a",
            },
        )

        hints = StructuredRequestHints.from_raw(
            request.sglang_hints, request.metadata
        )

        self.assertIsNone(hints.prefix_key)
        self.assertIsNone(hints.deadline_ms)
        self.assertEqual(hints.cache_affinity_key, "worker-group-a")

    def test_dependency_hints_are_normalized(self):
        hints = StructuredRequestHints.from_raw(
            {
                "latest_start_ms": 500,
                "internal_latest_start_us": 125_000,
                "downstream_release_credit": {
                    "expected_ready_count": 2,
                    "downstream_prefix_len": 80,
                    "release_gain_ms": 25,
                    "downstream_stage_id": "stage-b",
                    "ignored": "value",
                },
            }
        )

        self.assertEqual(hints.latest_start_ms, 500.0)
        self.assertEqual(hints.internal_latest_start_ms, 125.0)
        self.assertEqual(
            hints.downstream_release_credit,
            {
                "release_gain_ms": 25.0,
                "expected_ready_count": 2.0,
                "downstream_prefix_len": 80.0,
                "downstream_stage_id": "stage-b",
            },
        )

    def test_generate_req_input_expands_hints_for_batch(self):
        request = GenerateReqInput(
            text=["a", "b"],
            sampling_params={"max_new_tokens": 1},
            structured_hints={"prefix_key": "shared-prefix"},
        )

        request.normalize_batch_and_arguments()

        self.assertFalse(request.is_single)
        self.assertEqual(len(request.structured_hints), 2)
        self.assertEqual(request[0].structured_hints.prefix_key, "shared-prefix")
        self.assertEqual(request[1].structured_hints.prefix_key, "shared-prefix")


if __name__ == "__main__":
    unittest.main()
