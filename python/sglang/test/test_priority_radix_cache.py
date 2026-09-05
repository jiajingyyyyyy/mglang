import time
import unittest

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.managers.io_struct import StructuredRequestHints


class FakeAllocator:
    device = "cpu"

    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.append(value.tolist())
        return None

    def available_size(self):
        return 0


class PriorityRadixCacheTest(unittest.TestCase):
    def make_cache(self) -> tuple[RadixCache, FakeAllocator]:
        allocator = FakeAllocator()
        cache = RadixCache.create_simulated(
            mock_allocator=allocator,
            eviction_policy="priority",
        )
        return cache, allocator

    def test_priority_propagates_to_shared_ancestor(self) -> None:
        cache, _allocator = self.make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                priority=4.0,
                cache_pin_expires_at=time.monotonic() + 10.0,
                cache_hint_prefix_key="shared",
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 5, 6]),
                priority=2.0,
                cache_pin_expires_at=time.monotonic() + 10.0,
                cache_hint_prefix_key="shared",
            )
        )

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 9])))

        self.assertGreaterEqual(match.last_device_node.effective_priority(), 4.0)

    def test_ttl_expired_priority_becomes_zero(self) -> None:
        cache, _allocator = self.make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                priority=5.0,
                cache_pin_expires_at=time.monotonic() - 1.0,
                cache_hint_prefix_key="expired",
            )
        )

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))

        self.assertEqual(match.last_device_node.effective_priority(), 0.0)

    def test_priority_protected_blocks_counts_inserted_nodes(self) -> None:
        cache, _allocator = self.make_cache()

        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                priority=5.0,
                cache_pin_expires_at=time.monotonic() + 10.0,
                cache_hint_prefix_key="protected",
            )
        )

        stats = cache.priority_eviction_stats()
        self.assertGreater(stats["priority_protected_blocks"], 0.0)

    def test_structured_hints_preserve_cache_pin_ranges(self) -> None:
        hints = StructuredRequestHints.from_raw(
            {
                "priority": 3.0,
                "cache_pin_ranges": [
                    {
                        "start": 1,
                        "end": 3,
                        "priority": 7.0,
                        "cache_pin_ttl_ms": 1000.0,
                        "source": "reuse",
                        "release_after_hits": 1,
                        "release_consumer_key": "successor",
                        "lease_key": "predecessor->successor",
                    }
                ],
            }
        )

        self.assertIsNotNone(hints)
        self.assertEqual(hints.cache_pin_ranges[0]["start"], 1)
        self.assertEqual(hints.cache_pin_ranges[0]["end"], 3)
        self.assertEqual(hints.cache_pin_ranges[0]["source"], "reuse")
        self.assertEqual(hints.cache_pin_ranges[0]["release_after_hits"], 1.0)
        self.assertEqual(hints.cache_pin_ranges[0]["release_consumer_key"], "successor")
        self.assertEqual(
            hints.cache_pin_ranges[0]["lease_key"], "predecessor->successor"
        )

    def test_cross_request_reuse_releases_range_pin(self) -> None:
        class FakeReq:
            rid = "successor"
            cache_hint_consumer_key = "successor"

        cache, _allocator = self.make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                cache_pin_ranges=[
                    {
                        "start": 0,
                        "end": 4,
                        "priority": 5.0,
                        "cache_pin_ttl_ms": 10000.0,
                        "source": "release",
                        "release_after_hits": 1,
                        "release_consumer_key": "successor",
                        "lease_key": "predecessor->successor",
                    }
                ],
                cache_pin_metadata={"pin_mode": "layered_static"},
            )
        )

        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9]), req=FakeReq())
        )

        self.assertEqual(match.last_device_node.effective_priority(), 0.0)
        stats = cache.priority_eviction_stats()
        self.assertEqual(stats["pin_released_after_reuse_blocks"], 4.0)
        self.assertEqual(stats["pin_release_after_reuse_events"], 1.0)

    def test_non_consumer_and_read_only_matches_do_not_release_range_pin(self) -> None:
        class ParentReq:
            cache_hint_consumer_key = "predecessor"

        class UnrelatedReq:
            cache_hint_consumer_key = "unrelated"

        cache, _allocator = self.make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                cache_pin_ranges=[
                    {
                        "start": 0,
                        "end": 4,
                        "priority": 5.0,
                        "cache_pin_ttl_ms": 10000.0,
                        "source": "release",
                        "release_after_hits": 1,
                        "release_consumer_key": "successor",
                        "lease_key": "predecessor->successor",
                    }
                ],
                cache_pin_metadata={"pin_mode": "layered_static"},
            )
        )

        cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9])))
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9]), req=UnrelatedReq())
        )
        parent_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9]), req=ParentReq())
        )

        self.assertGreater(parent_match.last_device_node.effective_priority(), 0.0)
        stats = cache.priority_eviction_stats()
        self.assertEqual(stats["pin_released_after_reuse_blocks"], 0.0)

    def test_shared_prefix_keeps_other_dependency_lease_active(self) -> None:
        class FirstChild:
            rid = "child-a"
            cache_hint_consumer_key = "child-a"

        class SecondChild:
            rid = "child-b"
            cache_hint_consumer_key = "child-b"

        cache, _allocator = self.make_cache()
        for parent, child in (("parent-a", "child-a"), ("parent-b", "child-b")):
            cache.insert(
                InsertParams(
                    key=RadixKey([1, 2, 3, 4]),
                    cache_pin_ranges=[
                        {
                            "start": 0,
                            "end": 4,
                            "priority": 5.0,
                            "cache_pin_ttl_ms": 10000.0,
                            "source": "release",
                            "release_after_hits": 1,
                            "release_consumer_key": child,
                            "lease_key": f"{parent}->{child}",
                        }
                    ],
                    cache_pin_metadata={"pin_mode": "layered_static"},
                )
            )

        first_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9]), req=FirstChild())
        )
        self.assertGreater(first_match.last_device_node.effective_priority(), 0.0)
        first_stats = cache.priority_eviction_stats()
        self.assertEqual(first_stats["pin_lease_consumed_count"], 1.0)
        self.assertEqual(first_stats["pin_unique_lease_created_count"], 2.0)
        self.assertEqual(first_stats["pin_unique_lease_consumed_count"], 1.0)
        self.assertEqual(first_stats["pin_unique_lease_hit_rate"], 0.5)
        self.assertEqual(first_stats["pin_lease_active_count"], 1.0)
        self.assertEqual(first_stats["pin_released_after_reuse_blocks"], 0.0)

        second_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9]), req=SecondChild())
        )
        self.assertEqual(second_match.last_device_node.effective_priority(), 0.0)
        final_stats = cache.priority_eviction_stats()
        self.assertEqual(final_stats["pin_lease_consumed_count"], 2.0)
        self.assertEqual(final_stats["pin_unique_lease_created_count"], 2.0)
        self.assertEqual(final_stats["pin_unique_lease_consumed_count"], 2.0)
        self.assertEqual(final_stats["pin_unique_lease_hit_rate"], 1.0)
        self.assertEqual(final_stats["pin_lease_active_count"], 0.0)
        self.assertEqual(final_stats["pin_released_after_reuse_blocks"], 4.0)

    def test_cache_pin_ranges_protect_only_selected_tokens(self) -> None:
        cache, _allocator = self.make_cache()
        now = time.monotonic()

        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                priority=0.0,
                cache_pin_expires_at=now + 10.0,
                cache_pin_ranges=[
                    {
                        "start": 1,
                        "end": 4,
                        "priority": 5.0,
                        "cache_pin_ttl_ms": 10000.0,
                        "source": "reuse",
                    }
                ],
            )
        )

        stats = cache.priority_eviction_stats()
        self.assertEqual(stats["priority_protected_blocks"], 3.0)
        self.assertEqual(stats["priority_reuse_protected_blocks"], 3.0)
        self.assertEqual(stats["priority_request_protected_blocks"], 0.0)

        cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 9])))
        stats = cache.priority_eviction_stats()
        self.assertEqual(stats["reuse_after_pin_tokens"], 3.0)
        self.assertEqual(stats["reuse_after_pin_token_rate"], 1.0)

    def test_request_ttl_starts_when_inserted_into_radix_cache(self) -> None:
        class FakeReq:
            cache_priority = 5.0
            cache_pin_ttl_s = 0.2
            cache_pin_expires_at = time.monotonic() - 10.0

        before_insert = time.monotonic()
        expires_at = RadixCache._cache_pin_expires_at_for_req(FakeReq())

        self.assertIsNotNone(expires_at)
        self.assertGreater(expires_at, before_insert)

    def test_priority_evicts_lowest_priority_first(self) -> None:
        cache, allocator = self.make_cache()
        now = time.monotonic()
        cache.insert(
            InsertParams(
                key=RadixKey([10, 11, 12]),
                priority=5.0,
                cache_pin_expires_at=now + 10.0,
                cache_hint_prefix_key="hot",
            )
        )
        cache.insert(InsertParams(key=RadixKey([20, 21, 22]), priority=0.0))
        cache.insert(
            InsertParams(
                key=RadixKey([30, 31, 32]),
                priority=1.0,
                cache_pin_expires_at=now + 10.0,
                cache_hint_prefix_key="warm",
            )
        )

        cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(allocator.freed[0], [20, 21, 22])

    def test_pressure_demotes_low_value_request_pin_before_range_pin(self) -> None:
        cache, allocator = self.make_cache()
        now = time.monotonic()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(100, 120))),
                priority=1.0,
                cache_pin_expires_at=now + 10.0,
                cache_hint_prefix_key="broad-request",
                cache_pin_metadata={
                    "pin_mode": "request_soft_priority",
                    "motif_id": "motif-a",
                    "stage_id": "stage-a",
                },
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(200, 220))),
                priority=0.0,
                cache_pin_expires_at=now + 10.0,
                cache_pin_ranges=[
                    {
                        "start": 0,
                        "end": 20,
                        "priority": 1.0,
                        "cache_pin_ttl_ms": 10000.0,
                        "source": "reuse",
                    }
                ],
                cache_pin_metadata={
                    "pin_mode": "layered_static",
                    "motif_id": "motif-b",
                    "stage_id": "stage-b",
                },
            )
        )

        from sglang.srt.mem_cache import radix_cache as radix_cache_module

        old_pressure = radix_cache_module.PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION
        old_min_value = radix_cache_module.PRIORITY_PIN_DEMOTION_MIN_VALUE
        try:
            radix_cache_module.PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION = 0.01
            radix_cache_module.PRIORITY_PIN_DEMOTION_MIN_VALUE = 0.1
            cache.evict(EvictParams(num_tokens=5))
        finally:
            radix_cache_module.PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION = old_pressure
            radix_cache_module.PRIORITY_PIN_DEMOTION_MIN_VALUE = old_min_value

        self.assertEqual(allocator.freed[0], list(range(100, 120)))
        stats = cache.priority_eviction_stats()
        self.assertEqual(stats["priority_pressure_demoted_blocks"], 20.0)
        self.assertEqual(stats["priority_reuse_evicted_blocks"], 0.0)
        self.assertEqual(stats["pinned_blocks_evicted_before_reuse"], 20.0)
        self.assertEqual(stats["pinned_evicted_before_reuse_rate"], 0.5)
        self.assertEqual(
            stats["eviction_victim_pin_mode"]["request_soft_priority"], 20.0
        )
        self.assertEqual(stats["eviction_victim_motif_id"]["motif-a"], 20.0)
        self.assertEqual(stats["eviction_victim_stage_id"]["stage-a"], 20.0)
        self.assertEqual(stats["eviction_reason"]["forced_unpin"], 20.0)
        self.assertEqual(stats["forced_unpin_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
