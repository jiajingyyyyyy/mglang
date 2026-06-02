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
                    }
                ],
            }
        )

        self.assertIsNotNone(hints)
        self.assertEqual(hints.cache_pin_ranges[0]["start"], 1)
        self.assertEqual(hints.cache_pin_ranges[0]["end"], 3)
        self.assertEqual(hints.cache_pin_ranges[0]["source"], "reuse")

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


if __name__ == "__main__":
    unittest.main()
