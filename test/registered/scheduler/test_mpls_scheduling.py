import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import StructuredRequestHints
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams


def _req(
    rid,
    token_ids,
    *,
    ladder,
    arrival=1.0,
    latest_start_ms=500.0,
    internal_latest_start_ms=None,
    deadline_ms=None,
    release_credit=None,
    static_prefix_len=None,
    saved_prefill_cost_ms=None,
):
    req = Req(
        rid,
        str(rid),
        token_ids,
        SamplingParams(),
        structured_hints=StructuredRequestHints(
            prefix_ladder=ladder,
            latest_start_ms=latest_start_ms,
            internal_latest_start_ms=internal_latest_start_ms,
            deadline_ms=deadline_ms,
            downstream_release_credit=release_credit,
            static_prefix_len=static_prefix_len,
            saved_prefill_cost_ms=saved_prefill_cost_ms,
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

    def test_mpls_slo_alias_uses_mpls_policy_path(self):
        policy = SchedulePolicy(
            "mpls_slo",
            RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

        self.assertEqual(policy.policy, CacheAwarePolicy.MPLS_SLO)

    def test_mpls_selects_prefix_gain_inside_lag_frontier(self):
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

    def test_mpls_release_gain_can_select_upstream_stage(self):
        upstream_ladder = [
            {"level": "stage", "prefix_hash": "upstream", "prefix_len": 10}
        ]
        downstream_ladder = [
            {"level": "stage", "prefix_hash": "downstream", "prefix_len": 100}
        ]
        waiting_queue = [
            _req(
                "upstream",
                [1, 10],
                ladder=upstream_ladder,
                release_credit={"release_gain_ms": 250},
            ),
            _req("downstream-0", [2, 20], ladder=downstream_ladder),
            _req("downstream-1", [2, 21], ladder=downstream_ladder),
        ]

        with patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=1.05):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual(waiting_queue[0].rid, "upstream")
        stats = policy.get_mpls_stats()
        self.assertEqual(stats["selected_release_gain_ms_avg"], 250.0)
        self.assertEqual(stats["selected_total_gain_ms_avg"], 250.0)

    def test_mpls_slo_tier_promotes_critical_feasible_request(self):
        local_ladder = [
            {"level": "stage", "prefix_hash": "local", "prefix_len": 100}
        ]
        urgent_ladder = [
            {"level": "stage", "prefix_hash": "urgent", "prefix_len": 10}
        ]
        waiting_queue = [
            _req(
                "local-0",
                [1, 10],
                ladder=local_ladder,
                deadline_ms=500,
            ),
            _req(
                "local-1",
                [1, 11],
                ladder=local_ladder,
                deadline_ms=500,
            ),
            _req(
                "urgent",
                [2, 20],
                ladder=urgent_ladder,
                deadline_ms=100,
            ),
        ]

        patches = [
            patch(
                "sglang.srt.managers.schedule_policy."
                "MPLS_SLO_SERVICE_MS_PER_TOKEN",
                0.1,
            ),
            patch(
                "sglang.srt.managers.schedule_policy."
                "MPLS_SLO_CRITICAL_SLACK_MS",
                100.0,
            ),
        ]
        with patches[0], patches[1], patch(
            "sglang.srt.managers.schedule_policy.time.perf_counter",
            return_value=1.05,
        ):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual(waiting_queue[0].rid, "urgent")
        stats = policy.get_mpls_stats()
        self.assertGreater(stats["slo_critical_feasible_candidate_rate"], 0.0)

    def test_mpls_slo_tier_demotes_hopeless_request(self):
        hopeless_ladder = [
            {"level": "stage", "prefix_hash": "hopeless", "prefix_len": 100}
        ]
        feasible_ladder = [
            {"level": "stage", "prefix_hash": "feasible", "prefix_len": 10}
        ]
        waiting_queue = [
            _req(
                "hopeless-0",
                [1, 10],
                ladder=hopeless_ladder,
                deadline_ms=10,
            ),
            _req(
                "hopeless-1",
                [1, 11],
                ladder=hopeless_ladder,
                deadline_ms=10,
            ),
            _req(
                "feasible",
                [2, 20],
                ladder=feasible_ladder,
                deadline_ms=500,
            ),
        ]

        with patch(
            "sglang.srt.managers.schedule_policy.MPLS_SLO_SERVICE_MS_PER_TOKEN",
            0.1,
        ), patch(
            "sglang.srt.managers.schedule_policy.time.perf_counter",
            return_value=1.2,
        ):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual(waiting_queue[0].rid, "feasible")
        stats = policy.get_mpls_stats()
        self.assertGreater(stats["slo_hopeless_candidate_rate"], 0.0)

    def test_mpls_internal_deadline_overrides_generic_latest_start(self):
        urgent_ladder = [
            {"level": "stage", "prefix_hash": "urgent", "prefix_len": 10}
        ]
        local_ladder = [
            {"level": "stage", "prefix_hash": "local", "prefix_len": 100}
        ]
        waiting_queue = [
            _req(
                "local-0",
                [1, 10],
                ladder=local_ladder,
                latest_start_ms=500,
            ),
            _req(
                "urgent",
                [2, 20],
                ladder=urgent_ladder,
                latest_start_ms=500,
                internal_latest_start_ms=50,
            ),
            _req(
                "local-1",
                [1, 11],
                ladder=local_ladder,
                latest_start_ms=500,
            ),
        ]

        with patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=1.06):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual(waiting_queue[0].rid, "urgent")

    def test_mpls_prefix_gain_uses_prefill_ms_per_token(self):
        fast_ladder = [
            {"level": "stage", "prefix_hash": "fast", "prefix_len": 80}
        ]
        slow_ladder = [
            {"level": "stage", "prefix_hash": "slow", "prefix_len": 50}
        ]
        waiting_queue = [
            _req(
                "fast-0",
                [1, 10],
                ladder=fast_ladder,
                static_prefix_len=80,
                saved_prefill_cost_ms=40,
            ),
            _req(
                "fast-1",
                [1, 11],
                ladder=fast_ladder,
                static_prefix_len=80,
                saved_prefill_cost_ms=40,
            ),
            _req(
                "slow-0",
                [2, 20],
                ladder=slow_ladder,
                static_prefix_len=50,
                saved_prefill_cost_ms=150,
            ),
            _req(
                "slow-1",
                [2, 21],
                ladder=slow_ladder,
                static_prefix_len=50,
                saved_prefill_cost_ms=150,
            ),
        ]

        with patch("sglang.srt.managers.schedule_policy.time.perf_counter", return_value=1.05):
            policy = self._policy()
            policy.calc_priority(waiting_queue)

        self.assertEqual([req.rid for req in waiting_queue[:2]], ["slow-0", "slow-1"])
        stats = policy.get_mpls_stats()
        self.assertEqual(stats["selected_total_gain_ms_avg"], 150.0)

    def test_mpls_unified_loss_aware_gain_blocks_bad_positive_loss(self):
        with patch(
            "sglang.srt.managers.schedule_policy."
            "MPLS_DFS_OVERLAY_POSITIVE_LOSS_MIN_RATIO",
            4.0,
        ):
            gain = SchedulePolicy._mpls_unified_loss_aware_gain_ms(
                total_gain_ms=300.0,
                current_dfs_loss_ms=1000.0,
                opportunity_net_gain_ms=-700.0,
                approx_zero_loss=False,
            )

        self.assertEqual(gain, 0.0)

    def test_mpls_unified_loss_aware_gain_keeps_zero_loss_bridge(self):
        gain = SchedulePolicy._mpls_unified_loss_aware_gain_ms(
            total_gain_ms=300.0,
            current_dfs_loss_ms=0.0,
            opportunity_net_gain_ms=300.0,
            approx_zero_loss=True,
        )

        self.assertEqual(gain, 300.0)

    def test_mpls_unified_loss_aware_gain_allows_large_positive_loss(self):
        patches = [
            patch(
                "sglang.srt.managers.schedule_policy."
                "MPLS_DFS_OVERLAY_POSITIVE_LOSS_MIN_RATIO",
                2.0,
            ),
            patch(
                "sglang.srt.managers.schedule_policy."
                "MPLS_DFS_OVERLAY_MIN_NET_GAIN_MS",
                0.0,
            ),
        ]
        with patches[0], patches[1]:
            gain = SchedulePolicy._mpls_unified_loss_aware_gain_ms(
                total_gain_ms=5000.0,
                current_dfs_loss_ms=1000.0,
                opportunity_net_gain_ms=4000.0,
                approx_zero_loss=False,
            )

        self.assertEqual(gain, 4000.0)


if __name__ == "__main__":
    unittest.main()
