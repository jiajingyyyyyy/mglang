from __future__ import annotations

import time
import unittest
from collections import defaultdict
from types import SimpleNamespace

from sglang.srt.managers import schedule_policy as schedule_policy_module
from sglang.srt.managers.schedule_policy import SchedulePolicy


class FakeReq:
    def __init__(
        self,
        rid: str,
        *,
        level: str,
        prefix_hash: str,
        prefix_len: int,
        downstream_release_credit: dict | None = None,
    ) -> None:
        self.rid = rid
        self.extra_key = None
        self.lora_id = None
        self.grammar_key = None
        self.prefix_indices = [0] * prefix_len
        self.time_stats = SimpleNamespace(wait_queue_entry_time=time.perf_counter())
        self.structured_hints = SimpleNamespace(
            grammar_id=None,
            max_tokens_bucket=None,
            prefix_ladder=[
                {
                    "level": level,
                    "prefix_hash": prefix_hash,
                    "prefix_len": prefix_len,
                }
            ],
            downstream_release_credit=downstream_release_credit,
            saved_prefill_cost_ms=float(prefix_len),
            static_prefix_len=float(prefix_len),
            latest_start_ms=100_000.0,
        )


class TestSchedulePolicyMPLS(unittest.TestCase):
    def test_bridge_release_stats_track_future_ladder_waiting_queue_hit(self):
        upstream = FakeReq(
            "upstream",
            level="agent",
            prefix_hash="agent1-question",
            prefix_len=40,
            downstream_release_credit={
                "expected_ready_count": 1,
                "confidence": 1.0,
                "downstream_prefix_ladder": [
                    {
                        "level": "agent",
                        "prefix_hash": "agent2-question",
                        "prefix_len": 80,
                    }
                ],
            },
        )
        ready_downstream = FakeReq(
            "ready-downstream",
            level="agent",
            prefix_hash="agent2-question",
            prefix_len=80,
        )
        unrelated = FakeReq(
            "unrelated",
            level="agent",
            prefix_hash="agent3-question",
            prefix_len=120,
        )

        waiting_queue = [ready_downstream, unrelated, upstream]
        stats = defaultdict(float)

        SchedulePolicy._sort_by_mpls(waiting_queue, set(), stats)
        report = SchedulePolicy.__new__(SchedulePolicy)
        report.mpls_stats = stats
        mpls_stats = report.get_mpls_stats()

        self.assertEqual(waiting_queue[0].rid, "upstream")
        self.assertEqual(stats["future_ladder_req_count_total"], 1.0)
        self.assertEqual(stats["future_ladder_entry_count_total"], 1.0)
        self.assertEqual(stats["future_ladder_match_count_total"], 1.0)
        self.assertEqual(stats["bridge_hit_req_count_total"], 1.0)
        self.assertEqual(stats["selected_with_future_ladder_count"], 1.0)
        self.assertEqual(stats["selected_with_future_ladder_match_count"], 1.0)
        self.assertEqual(stats["selected_with_bridge_gain_count"], 1.0)
        self.assertGreater(stats["selected_bridge_release_gain_ms_total"], 0.0)
        self.assertEqual(mpls_stats["future_ladder_entry_match_rate"], 1.0)
        self.assertEqual(mpls_stats["selected_bridge_gain_rate"], 1.0)

    def test_imminent_unlock_selects_future_ladder_when_enabled(self):
        upstream = FakeReq(
            "upstream",
            level="agent",
            prefix_hash="agent1-question",
            prefix_len=40,
            downstream_release_credit={
                "expected_ready_count": 1,
                "confidence": 1.0,
                "future_prefix_ladder": [
                    {
                        "level": "agent",
                        "prefix_hash": "agent2-question",
                        "prefix_len": 80,
                    }
                ],
            },
        )
        longer_unrelated = FakeReq(
            "longer-unrelated",
            level="agent",
            prefix_hash="agent3-question",
            prefix_len=120,
        )

        waiting_queue = [longer_unrelated, upstream]
        stats = defaultdict(float)

        old_weight = schedule_policy_module.MPLS_IMMINENT_UNLOCK_WEIGHT
        schedule_policy_module.MPLS_IMMINENT_UNLOCK_WEIGHT = 0.5
        try:
            SchedulePolicy._sort_by_mpls(waiting_queue, set(), stats)
        finally:
            schedule_policy_module.MPLS_IMMINENT_UNLOCK_WEIGHT = old_weight
        report = SchedulePolicy.__new__(SchedulePolicy)
        report.mpls_stats = stats
        mpls_stats = report.get_mpls_stats()

        self.assertEqual(waiting_queue[0].rid, "upstream")
        self.assertEqual(stats["future_ladder_req_count_total"], 1.0)
        self.assertEqual(stats["future_ladder_match_count_total"], 0.0)
        self.assertEqual(stats["bridge_hit_req_count_total"], 0.0)
        self.assertEqual(stats["imminent_unlock_req_count_total"], 1.0)
        self.assertEqual(stats["selected_with_imminent_unlock_count"], 1.0)
        self.assertGreater(stats["selected_imminent_unlock_gain_ms_total"], 0.0)
        self.assertEqual(mpls_stats["selected_imminent_unlock_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
