import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import StructuredRequestHints
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams


def _req(rid, token_ids, *, ladder, arrival=1.0, latest_start_ms=500.0):
    req = Req(
        rid,
        str(rid),
        token_ids,
        SamplingParams(),
        structured_hints=StructuredRequestHints(
            prefix_ladder=ladder,
            latest_start_ms=latest_start_ms,
            grammar_id="freeform",
            max_tokens_bucket="le256",
        ),
    )
    req.time_stats.wait_queue_entry_time = arrival
    return req


class TestMplsScheduling(unittest.TestCase):
    def _policy(self):
        policy = SchedulePolicy(
            "mpls",
            RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.MPLS)
        return policy

    def test_mpls_selects_prefix_gain_inside_topk_lag_frontier(self):
        long_ladder = [
            {"level": "stage_tool", "prefix_hash": "long", "prefix_len": 100}
        ]
        short_ladder = [
            {"level": "stage_tool", "prefix_hash": "short", "prefix_len": 20}
        ]
        waiting_queue = [
            _req("short-0", [1, 10], ladder=short_ladder),
            _req("long-0", [2, 20], ladder=long_ladder),
            _req("short-1", [1, 11], ladder=short_ladder),
            _req("long-1", [2, 21], ladder=long_ladder),
            _req("short-2", [1, 12], ladder=short_ladder),
        ]

        with patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=1.05):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual([req.rid for req in waiting_queue[:2]], ["long-0", "long-1"])
        stats = policy.get_mpls_stats()
        self.assertEqual(stats["selected_count"], 1.0)
        self.assertEqual(stats["selected_prefix_len_avg"], 100.0)
        self.assertEqual(stats["selected_prefix_gain_avg"], 100.0)

    def test_mpls_expired_lease_overrides_larger_prefix_gain(self):
        urgent_ladder = [
            {"level": "stage", "prefix_hash": "urgent", "prefix_len": 10}
        ]
        local_ladder = [
            {"level": "stage", "prefix_hash": "local", "prefix_len": 100}
        ]
        waiting_queue = [
            _req("local-0", [1, 10], ladder=local_ladder, latest_start_ms=500),
            _req("urgent", [2, 20], ladder=urgent_ladder, latest_start_ms=50),
            _req("local-1", [1, 11], ladder=local_ladder, latest_start_ms=500),
            _req("local-2", [1, 12], ladder=local_ladder, latest_start_ms=500),
        ]

        with patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=1.06):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual(waiting_queue[0].rid, "urgent")
        stats = policy.get_mpls_stats()
        self.assertEqual(stats["selected_expired_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
