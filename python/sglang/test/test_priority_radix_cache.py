import time
import unittest

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class FakeAllocator:
    device = "cpu"

    def free(self, value):
        return None

    def available_size(self):
        return 0


class PriorityRadixCacheTest(unittest.TestCase):
    def make_cache(self) -> RadixCache:
        cache = RadixCache.create_simulated(mock_allocator=FakeAllocator())
        cache.eviction_policy = "priority"
        return cache

    def test_priority_propagates_to_shared_ancestor(self) -> None:
        cache = self.make_cache()
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
        cache = self.make_cache()
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


if __name__ == "__main__":
    unittest.main()
