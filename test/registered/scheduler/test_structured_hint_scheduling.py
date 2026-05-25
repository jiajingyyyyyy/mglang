import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import StructuredRequestHints
from sglang.srt.managers.schedule_policy import SchedulePolicy


class _DummyTreeCache:
    disable = True


def _req(rid, prefix_key=None, arrival=0.0, priority=0):
    return SimpleNamespace(
        rid=rid,
        priority=priority,
        structured_hints=(
            StructuredRequestHints(prefix_key=prefix_key)
            if prefix_key is not None
            else None
        ),
        time_stats=SimpleNamespace(wait_queue_entry_time=arrival),
    )


class TestStructuredHintScheduling(unittest.TestCase):
    def _policy(self):
        return SchedulePolicy(
            "structured-hint",
            _DummyTreeCache(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def test_interleaved_prefixes_are_grouped_by_anchor_prefix(self):
        waiting_queue = [
            _req("a0", "A", 0),
            _req("b0", "B", 1),
            _req("c0", "C", 2),
            _req("a1", "A", 3),
            _req("b1", "B", 4),
            _req("c1", "C", 5),
        ]

        self._policy().calc_priority(
            waiting_queue, running_batch=SimpleNamespace(reqs=[])
        )

        self.assertEqual([req.rid for req in waiting_queue[:2]], ["a0", "a1"])
        self.assertCountEqual(
            [req.rid for req in waiting_queue],
            ["a0", "b0", "c0", "a1", "b1", "c1"],
        )

    def test_hot_prefix_is_capped_and_balanced_with_other_prefixes(self):
        waiting_queue = [
            _req("a0", "A", 0),
            _req("b0", "B", 1),
            _req("a1", "A", 2),
            _req("b1", "B", 3),
            _req("a2", "A", 4),
            _req("b2", "B", 5),
            _req("a3", "A", 6),
            _req("b3", "B", 7),
            _req("a4", "A", 8),
            _req("b4", "B", 9),
        ]

        with patch("sglang.srt.managers.schedule_policy.STRUCTURED_HINT_GROUP_CAP", 2):
            self._policy().calc_priority(
                waiting_queue, running_batch=SimpleNamespace(reqs=[])
            )

        self.assertEqual(
            [req.rid for req in waiting_queue],
            ["a0", "a1", "b0", "b1", "a2", "a3", "b2", "b3", "a4", "b4"],
        )

    def test_without_hints_structured_hint_policy_is_fcfs(self):
        waiting_queue = [
            _req("r0", None, 0),
            _req("r1", None, 1),
            _req("r2", None, 2),
        ]

        self._policy().calc_priority(
            waiting_queue, running_batch=SimpleNamespace(reqs=[])
        )

        self.assertEqual([req.rid for req in waiting_queue], ["r0", "r1", "r2"])

    def test_running_batch_hint_is_used_as_anchor(self):
        waiting_queue = [
            _req("a0", "A", 0),
            _req("b0", "B", 1),
            _req("b1", "B", 2),
            _req("a1", "A", 3),
        ]
        running_batch = SimpleNamespace(reqs=[_req("running-b", "B", 0)])

        self._policy().calc_priority(waiting_queue, running_batch=running_batch)

        self.assertEqual([req.rid for req in waiting_queue[:2]], ["b0", "b1"])


if __name__ == "__main__":
    unittest.main()
