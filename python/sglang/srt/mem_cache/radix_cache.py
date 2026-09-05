from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

PRIORITY_PIN_PRESSURE_DEMOTION = (
    os.environ.get("SGLANG_PRIORITY_PIN_PRESSURE_DEMOTION", "true").lower()
    in ("1", "true", "yes", "on")
)
PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION = float(
    os.environ.get("SGLANG_PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION", "0.25")
)
PRIORITY_PIN_DEMOTION_MIN_VALUE = float(
    os.environ.get("SGLANG_PRIORITY_PIN_DEMOTION_MIN_VALUE", "0.05")
)
PRIORITY_DYNAMIC_PIN_DEMOTION_MIN_UTILITY = float(
    os.environ.get("SGLANG_PRIORITY_DYNAMIC_PIN_DEMOTION_MIN_UTILITY", "0.05")
)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: float = 0.0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = float(priority or 0.0)
        self.cache_pin_expires_at: Optional[float] = None
        self.cache_hint_prefix_key: Optional[str] = None
        self.cache_pin_source: Optional[str] = None
        self.cache_pin_mode: Optional[str] = None
        self.cache_hint_agent_type: Optional[str] = None
        self.cache_hint_motif_id: Optional[str] = None
        self.cache_hint_stage_id: Optional[str] = None
        self.cache_pin_utility: float = 0.0
        self.cache_pin_saved_prefill_ms: float = 0.0
        self.cache_pin_structure_release_gain_ms: float = 0.0
        self.cache_pin_expected_queue_saving_ms: float = 0.0
        self.cache_pin_dynamic: bool = False
        self.cache_pin_forced_unpin = False
        self.cache_pin_released_after_reuse = False
        self.cache_pin_leases: dict[str, dict[str, Any]] = {}
        self.cache_hit_after_pin = 0

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def effective_priority(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        lease_priority = max(
            (
                max(0.0, float(lease.get("priority") or 0.0))
                for lease in self.cache_pin_leases.values()
                if int(lease.get("remaining_hits") or 0) > 0
                and (
                    lease.get("expires_at") is None or now <= float(lease["expires_at"])
                )
            ),
            default=0.0,
        )
        legacy_priority = max(0.0, float(self.priority or 0.0))
        if self.cache_pin_expires_at is not None:
            if now > self.cache_pin_expires_at:
                legacy_priority = 0.0
        return max(legacy_priority, lease_priority)

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
            eviction_policy=eviction_policy,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.priority_protected_blocks = 0
        self.priority_expired_blocks = 0
        self.priority_evicted_blocks = 0
        self.priority_pressure_demoted_blocks = 0
        self.priority_pressure_demotion_events = 0
        self.pin_demoted_under_pressure_blocks = 0
        self.pin_demoted_without_pressure_blocks = 0
        self.forced_unpin_count = 0
        self.pinned_blocks_total = 0
        self.pinned_blocks_evicted = 0
        self.pinned_blocks_evicted_before_reuse = 0
        self.pinned_blocks_reused = 0
        self.cold_prefill_tokens_after_eviction = 0
        self.protected_but_not_reused_blocks = 0
        self.evicted_hot_prefix_count = 0
        self.reuse_after_pin_count = 0
        self.reuse_after_pin_events = 0
        self.reuse_after_pin_tokens = 0
        self.reuse_after_pin_node_hits = 0
        self.pin_released_after_reuse_blocks = 0
        self.pin_release_after_reuse_events = 0
        self.pin_lease_created_count = 0
        self.pin_lease_refreshed_count = 0
        self.pin_lease_consumed_count = 0
        # Radix-node splits copy lease metadata, so node-level counters do not
        # equal the number of logical retention hints.
        self.pin_unique_lease_created_keys: set[str] = set()
        self.pin_unique_lease_consumed_keys: set[str] = set()
        self.pin_unique_lease_expires_at: dict[str, Optional[float]] = {}
        self.priority_protected_blocks_by_source = defaultdict(float)
        self.priority_evicted_blocks_by_source = defaultdict(float)
        self.protected_blocks_by_agent_type = defaultdict(float)
        self.reuse_tokens_by_agent_type = defaultdict(float)
        self.protected_but_not_reused_by_agent_type = defaultdict(float)
        self.priority_evicted_blocks_by_agent_type = defaultdict(float)
        self.eviction_victim_pin_mode = defaultdict(float)
        self.eviction_victim_motif_id = defaultdict(float)
        self.eviction_victim_stage_id = defaultdict(float)
        self.eviction_reason = defaultdict(float)
        self.free_blocks_ratio_ema = 0.0
        self.eviction_rate_ema = 0.0
        self.free_blocks_ratio_samples = []
        self.estimated_saved_prefill_ms = 0.0
        self.actual_cached_prefill_tokens = 0.0
        self.actual_prefill_tokens = 0.0
        self._last_eviction_sample_time = None
        self.evictable_leaves.clear()
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node, matched_nodes = self._match_prefix_helper(self.root_node, key)
        pinned_token_hits = 0
        pinned_node_hits = 0
        agent_hits = defaultdict(float)
        saved_prefill_ms = 0.0
        release_candidates = []
        consumer_key = str(getattr(params.req, "cache_hint_consumer_key", None) or "")
        now = time.monotonic()
        for matched_node in matched_nodes:
            if matched_node.effective_priority(now) > 0:
                pinned_node_hits += 1
                node_tokens = self._node_value_len(matched_node)
                pinned_token_hits += node_tokens
                agent_hits[self._agent_bucket(matched_node)] += node_tokens
                saved_prefill_ms += float(
                    getattr(matched_node, "cache_pin_saved_prefill_ms", 0.0) or 0.0
                )
                matched_node.cache_hit_after_pin += 1
                matching_leases = [
                    lease_key
                    for lease_key, lease in matched_node.cache_pin_leases.items()
                    if consumer_key
                    and consumer_key == str(lease.get("consumer_key") or "")
                    and int(lease.get("remaining_hits") or 0) > 0
                    and (
                        lease.get("expires_at") is None
                        or now <= float(lease["expires_at"])
                    )
                ]
                if matching_leases:
                    release_candidates.append(
                        (matched_node, node_tokens, matching_leases)
                    )
        if last_node is not self.root_node and last_node.effective_priority() > 0:
            self.reuse_after_pin_count += 1
        if pinned_token_hits > 0:
            self.reuse_after_pin_events += 1
            self.reuse_after_pin_tokens += pinned_token_hits
            self.reuse_after_pin_node_hits += pinned_node_hits
            self.pinned_blocks_reused += pinned_token_hits
            self.actual_cached_prefill_tokens += pinned_token_hits
            self.estimated_saved_prefill_ms += saved_prefill_ms
            for agent_type, token_count in agent_hits.items():
                self.reuse_tokens_by_agent_type[agent_type] += token_count
        released_blocks = 0
        for matched_node, node_tokens, matching_leases in release_candidates:
            priority_before = matched_node.effective_priority(now)
            for lease_key in matching_leases:
                lease = matched_node.cache_pin_leases[lease_key]
                lease["remaining_hits"] = max(
                    0, int(lease.get("remaining_hits") or 0) - 1
                )
                if lease["remaining_hits"] == 0:
                    self.pin_lease_consumed_count += 1
                    self.pin_unique_lease_consumed_keys.add(lease_key)
            if priority_before > 0 and matched_node.effective_priority(now) <= 0:
                matched_node.cache_pin_released_after_reuse = True
                released_blocks += node_tokens
        if released_blocks > 0:
            self.pin_released_after_reuse_blocks += released_blocks
            self.pin_release_after_reuse_events += 1
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(
            self.root_node,
            key,
            value,
            priority,
            params.cache_pin_expires_at,
            params.cache_hint_prefix_key,
            params.cache_pin_ranges,
            params.cache_pin_metadata,
        )
        return InsertResult(prefix_len=prefix_len)

    @staticmethod
    def _cache_pin_expires_at_for_req(req: Req) -> Optional[float]:
        ttl_s = getattr(req, "cache_pin_ttl_s", None)
        priority = getattr(req, "cache_priority", 0.0) or 0.0
        if priority > 0.0 and ttl_s is not None:
            try:
                ttl_s = max(0.0, float(ttl_s))
            except (TypeError, ValueError):
                ttl_s = 0.0
            if ttl_s > 0.0:
                return time.monotonic() + ttl_s
        return getattr(req, "cache_pin_expires_at", None)

    @staticmethod
    def _cache_pin_ranges_for_req(req: Req) -> list[dict[str, Any]]:
        ranges = getattr(req, "cache_pin_ranges", None)
        if not isinstance(ranges, list):
            return []
        return [row for row in ranges if isinstance(row, dict)]

    @staticmethod
    def _cache_pin_metadata_for_req(req: Req) -> dict[str, str]:
        ranges = RadixCache._cache_pin_ranges_for_req(req)
        pin_mode = getattr(req, "cache_hint_pin_mode", None)
        if not pin_mode:
            pin_mode = "layered_static" if ranges else "request_soft_priority"
        return {
            "pin_mode": str(pin_mode),
            "motif_id": str(getattr(req, "cache_hint_motif_id", None) or "unknown"),
            "stage_id": str(getattr(req, "cache_hint_stage_id", None) or "unknown"),
            "agent_type": str(getattr(req, "cache_hint_agent_type", None) or "unknown"),
            "saved_prefill_ms": float(
                getattr(req, "cache_hint_saved_prefill_ms", None)
                or getattr(req, "cache_priority", 0.0)
                or 0.0
            ),
        }

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "cache_priority", 0.0) or 0.0
            cache_pin_ranges = self._cache_pin_ranges_for_req(req)
            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    priority=0.0 if cache_pin_ranges else priority,
                    cache_pin_expires_at=self._cache_pin_expires_at_for_req(req),
                    cache_hint_prefix_key=getattr(req, "cache_hint_prefix_key", None),
                    cache_pin_ranges=cache_pin_ranges,
                    cache_pin_metadata=self._cache_pin_metadata_for_req(req),
                )
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=(
                    0.0
                    if self._cache_pin_ranges_for_req(req)
                    else getattr(req, "cache_priority", 0.0) or 0.0
                ),
                cache_pin_expires_at=self._cache_pin_expires_at_for_req(req),
                cache_hint_prefix_key=getattr(req, "cache_hint_prefix_key", None),
                cache_pin_ranges=self._cache_pin_ranges_for_req(req),
                cache_pin_metadata=self._cache_pin_metadata_for_req(req),
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        self._maybe_demote_low_value_request_pins(num_tokens)
        self._sample_eviction_pressure(num_tokens, 0, start_time)
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x.priority > 0 or x.cache_pin_leases:
                source = str(getattr(x, "cache_pin_source", None) or "request")
                node_tokens = len(x.value)
                pin_mode = str(getattr(x, "cache_pin_mode", None) or source)
                agent_type = self._agent_bucket(x)
                motif_id = str(getattr(x, "cache_hint_motif_id", None) or "unknown")
                stage_id = str(getattr(x, "cache_hint_stage_id", None) or "unknown")
                self.pinned_blocks_evicted += node_tokens
                self.eviction_victim_pin_mode[pin_mode] += node_tokens
                self.priority_evicted_blocks_by_agent_type[agent_type] += node_tokens
                self.eviction_victim_motif_id[motif_id] += node_tokens
                self.eviction_victim_stage_id[stage_id] += node_tokens
                if x.effective_priority() <= 0:
                    self.priority_expired_blocks += node_tokens
                    if getattr(x, "cache_pin_released_after_reuse", False):
                        reason = "reuse_consumed"
                    elif getattr(x, "cache_pin_forced_unpin", False):
                        reason = "forced_unpin"
                    else:
                        reason = "ttl_expired"
                    self.eviction_reason[reason] += node_tokens
                else:
                    self.priority_evicted_blocks += node_tokens
                    self.priority_evicted_blocks_by_source[source] += node_tokens
                    self.evicted_hot_prefix_count += 1
                    self.eviction_reason["memory_pressure"] += node_tokens
                if x.cache_hit_after_pin <= 0:
                    self.protected_but_not_reused_blocks += node_tokens
                    self.pinned_blocks_evicted_before_reuse += node_tokens
                    self.protected_but_not_reused_by_agent_type[agent_type] += node_tokens
                self.cold_prefill_tokens_after_eviction += node_tokens

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        self._sample_eviction_pressure(num_tokens, num_evicted, time.perf_counter())
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def priority_eviction_stats(self) -> dict[str, float]:
        total_protected = float(self.priority_protected_blocks or 0)
        source_protected = dict(self.priority_protected_blocks_by_source)
        source_evicted = dict(self.priority_evicted_blocks_by_source)
        agent_protected = dict(self.protected_blocks_by_agent_type)
        agent_reuse = dict(self.reuse_tokens_by_agent_type)
        agent_not_reused = dict(self.protected_but_not_reused_by_agent_type)
        agent_evicted = dict(self.priority_evicted_blocks_by_agent_type)
        free_ratio_avg = (
            sum(self.free_blocks_ratio_samples) / len(self.free_blocks_ratio_samples)
            if self.free_blocks_ratio_samples
            else self._free_blocks_ratio()
        )
        free_ratio_p5 = (
            self._percentile(self.free_blocks_ratio_samples, 0.05)
            if self.free_blocks_ratio_samples
            else self._free_blocks_ratio()
        )
        now = time.monotonic()
        leases = [
            lease
            for node in self._iter_tree_nodes(self.root_node)
            for lease in node.cache_pin_leases.values()
        ]
        active_leases = sum(
            1
            for lease in leases
            if int(lease.get("remaining_hits") or 0) > 0
            and (
                lease.get("expires_at") is None
                or now <= float(lease["expires_at"])
            )
        )
        expired_leases = sum(
            1
            for lease in leases
            if int(lease.get("remaining_hits") or 0) > 0
            and lease.get("expires_at") is not None
            and now > float(lease["expires_at"])
        )
        unique_created = len(self.pin_unique_lease_created_keys)
        unique_consumed = len(self.pin_unique_lease_consumed_keys)
        unique_active = sum(
            1
            for lease_key, expires_at in self.pin_unique_lease_expires_at.items()
            if lease_key not in self.pin_unique_lease_consumed_keys
            and (expires_at is None or now <= expires_at)
        )
        unique_expired = sum(
            1
            for lease_key, expires_at in self.pin_unique_lease_expires_at.items()
            if lease_key not in self.pin_unique_lease_consumed_keys
            and expires_at is not None
            and now > expires_at
        )
        return {
            "pinned_blocks_total": float(self.pinned_blocks_total),
            "pinned_blocks_evicted": float(self.pinned_blocks_evicted),
            "pinned_blocks_evicted_before_reuse": float(
                self.pinned_blocks_evicted_before_reuse
            ),
            "pinned_blocks_reused": float(self.pinned_blocks_reused),
            "pinned_blocks_not_reused": float(
                max(0.0, self.pinned_blocks_total - self.pinned_blocks_reused)
            ),
            "pinned_evicted_before_reuse_rate": (
                float(self.pinned_blocks_evicted_before_reuse)
                / float(self.pinned_blocks_total)
                if self.pinned_blocks_total > 0.0
                else 0.0
            ),
            "eviction_victim_pin_mode": dict(self.eviction_victim_pin_mode),
            "eviction_victim_motif_id": dict(self.eviction_victim_motif_id),
            "eviction_victim_stage_id": dict(self.eviction_victim_stage_id),
            "eviction_reason": dict(self.eviction_reason),
            "free_blocks_ratio_ema": float(self.free_blocks_ratio_ema),
            "free_blocks_ratio_avg": float(free_ratio_avg),
            "free_blocks_ratio_p5": float(free_ratio_p5),
            "eviction_rate_ema": float(self.eviction_rate_ema),
            "cold_prefill_tokens_after_eviction": float(
                self.cold_prefill_tokens_after_eviction
            ),
            "actual_prefill_tokens": float(self.actual_prefill_tokens),
            "actual_cached_prefill_tokens": float(self.actual_cached_prefill_tokens),
            "actual_recomputed_prefill_tokens": float(
                max(0.0, self.actual_prefill_tokens - self.actual_cached_prefill_tokens)
            ),
            "estimated_saved_prefill_ms": float(self.estimated_saved_prefill_ms),
            "forced_unpin_count": float(self.forced_unpin_count),
            "protected_but_not_reused_blocks": float(self.protected_but_not_reused_blocks),
            "evicted_hot_prefix_count": float(self.evicted_hot_prefix_count),
            "reuse_after_pin_rate": (
                float(self.reuse_after_pin_count) / total_protected
                if total_protected > 0.0
                else 0.0
            ),
            "reuse_after_pin_count": float(self.reuse_after_pin_count),
            "reuse_after_pin_events": float(self.reuse_after_pin_events),
            "reuse_after_pin_tokens": float(self.reuse_after_pin_tokens),
            "reuse_after_pin_node_hits": float(self.reuse_after_pin_node_hits),
            "pin_released_after_reuse_blocks": float(
                self.pin_released_after_reuse_blocks
            ),
            "pin_release_after_reuse_events": float(
                self.pin_release_after_reuse_events
            ),
            "pin_lease_created_count": float(self.pin_lease_created_count),
            "pin_lease_refreshed_count": float(self.pin_lease_refreshed_count),
            "pin_lease_consumed_count": float(self.pin_lease_consumed_count),
            "pin_lease_active_count": float(active_leases),
            "pin_lease_expired_count": float(expired_leases),
            "pin_unique_lease_created_count": float(unique_created),
            "pin_unique_lease_consumed_count": float(unique_consumed),
            "pin_unique_lease_active_count": float(unique_active),
            "pin_unique_lease_expired_count": float(unique_expired),
            "pin_unique_lease_hit_rate": (
                float(unique_consumed) / float(unique_created)
                if unique_created > 0
                else 0.0
            ),
            "reuse_after_pin_token_rate": (
                float(self.reuse_after_pin_tokens) / total_protected
                if total_protected > 0.0
                else 0.0
            ),
            "priority_protected_blocks": total_protected,
            "priority_expired_blocks": float(self.priority_expired_blocks),
            "priority_evicted_blocks": float(self.priority_evicted_blocks),
            "priority_pressure_demoted_blocks": float(
                self.priority_pressure_demoted_blocks
            ),
            "priority_pressure_demotion_events": float(
                self.priority_pressure_demotion_events
            ),
            "pin_demoted_under_pressure_blocks": float(
                self.pin_demoted_under_pressure_blocks
            ),
            "pin_demoted_without_pressure_blocks": float(
                self.pin_demoted_without_pressure_blocks
            ),
            "priority_request_protected_blocks": float(source_protected.get("request", 0.0)),
            "priority_reuse_protected_blocks": float(source_protected.get("reuse", 0.0)),
            "priority_release_protected_blocks": float(source_protected.get("release", 0.0)),
            "priority_range_protected_blocks": float(source_protected.get("range", 0.0)),
            "priority_request_evicted_blocks": float(source_evicted.get("request", 0.0)),
            "priority_reuse_evicted_blocks": float(source_evicted.get("reuse", 0.0)),
            "priority_release_evicted_blocks": float(source_evicted.get("release", 0.0)),
            "priority_range_evicted_blocks": float(source_evicted.get("range", 0.0)),
            "protected_blocks_by_react": float(agent_protected.get("react", 0.0)),
            "protected_blocks_by_motif": float(agent_protected.get("motif", 0.0)),
            "protected_blocks_by_unknown": float(agent_protected.get("unknown", 0.0)),
            "reuse_tokens_by_react": float(agent_reuse.get("react", 0.0)),
            "reuse_tokens_by_motif": float(agent_reuse.get("motif", 0.0)),
            "reuse_tokens_by_unknown": float(agent_reuse.get("unknown", 0.0)),
            "protected_but_not_reused_by_react": float(agent_not_reused.get("react", 0.0)),
            "protected_but_not_reused_by_motif": float(agent_not_reused.get("motif", 0.0)),
            "protected_but_not_reused_by_unknown": float(agent_not_reused.get("unknown", 0.0)),
            "priority_evicted_blocks_by_react": float(agent_evicted.get("react", 0.0)),
            "priority_evicted_blocks_by_motif": float(agent_evicted.get("motif", 0.0)),
            "priority_evicted_blocks_by_unknown": float(agent_evicted.get("unknown", 0.0)),
        }

    def available_and_evictable_str(self) -> str:
        base = super().available_and_evictable_str()
        stats = self.priority_eviction_stats()
        return (
            base
            + "Priority eviction stats: "
            + ", ".join(f"{key}={value}" for key, value in stats.items())
            + "\n"
        )

    def _maybe_demote_low_value_request_pins(self, requested_tokens: int) -> None:
        if (
            not PRIORITY_PIN_PRESSURE_DEMOTION
            or self.eviction_policy != "priority"
            or requested_tokens <= 0
            or self.evictable_size_ <= 0
        ):
            return

        pressure = float(requested_tokens) / max(1.0, float(self.evictable_size_))
        threshold = max(0.0, float(PRIORITY_PIN_DEMOTION_PRESSURE_FRACTION or 0.0))
        if pressure < threshold:
            return

        min_value = max(0.0, float(PRIORITY_PIN_DEMOTION_MIN_VALUE or 0.0))
        if min_value <= 0.0:
            return

        now = time.monotonic()
        demoted_blocks = 0
        for node in self._iter_tree_nodes(self.root_node):
            if node is self.root_node or node.evicted:
                continue
            pin_mode = str(getattr(node, "cache_pin_mode", None) or "")
            source = str(getattr(node, "cache_pin_source", None) or "")
            dynamic_guarded = pin_mode in {
                "generic_layered_dynamic",
                "structure_layered_dynamic",
                "motif_layered_dynamic",
                "request_soft_priority_guarded",
            } or bool(getattr(node, "cache_pin_dynamic", False))
            if source != "request" and not dynamic_guarded:
                continue
            if node.effective_priority(now) <= 0.0:
                continue
            block_count = max(1, self._node_value_len(node))
            hit_count = max(0, int(getattr(node, "cache_hit_after_pin", 0) or 0))
            marginal_value = node.effective_priority(now) * (1.0 + hit_count)
            marginal_value /= float(block_count)
            threshold = (
                max(min_value, PRIORITY_DYNAMIC_PIN_DEMOTION_MIN_UTILITY)
                if dynamic_guarded
                else min_value
            )
            if marginal_value >= threshold:
                continue
            node.cache_pin_expires_at = now - 1e-6
            node.cache_pin_forced_unpin = True
            demoted_blocks += block_count

        if demoted_blocks > 0:
            self.priority_pressure_demoted_blocks += demoted_blocks
            self.priority_pressure_demotion_events += 1
            self.pin_demoted_under_pressure_blocks += demoted_blocks
            self.forced_unpin_count += 1

    def _free_blocks_ratio(self) -> float:
        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        try:
            available = float(allocator.available_size())
            total = float(getattr(allocator, "size", 0.0) or 0.0)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, available / total)) if total > 0.0 else 0.0

    def _sample_eviction_pressure(
        self, requested_tokens: int, evicted_tokens: int, now: float
    ) -> None:
        ratio = self._free_blocks_ratio()
        self.free_blocks_ratio_samples.append(ratio)
        if len(self.free_blocks_ratio_samples) > 4096:
            self.free_blocks_ratio_samples = self.free_blocks_ratio_samples[-4096:]
        alpha = 0.2
        if self.free_blocks_ratio_ema <= 0.0:
            self.free_blocks_ratio_ema = ratio
        else:
            self.free_blocks_ratio_ema = alpha * ratio + (1.0 - alpha) * self.free_blocks_ratio_ema

        last = self._last_eviction_sample_time
        self._last_eviction_sample_time = now
        if last is None:
            return
        dt = max(1e-6, float(now) - float(last))
        rate = float(evicted_tokens if evicted_tokens > 0 else requested_tokens) / dt
        if self.eviction_rate_ema <= 0.0:
            self.eviction_rate_ema = rate
        else:
            self.eviction_rate_ema = alpha * rate + (1.0 - alpha) * self.eviction_rate_ema

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(max(0, min(len(ordered) - 1, round(percentile * (len(ordered) - 1)))))
        return float(ordered[index])

    def _iter_tree_nodes(self, node: TreeNode) -> Iterator[TreeNode]:
        yield node
        for child in list(node.children.values()):
            yield from self._iter_tree_nodes(child)

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        matched_nodes = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                matched_nodes.append(new_node)
                node = new_node
                break
            else:
                value.append(child.value)
                matched_nodes.append(child)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node, matched_nodes

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.cache_pin_expires_at = child.cache_pin_expires_at
        new_node.cache_hint_prefix_key = child.cache_hint_prefix_key
        new_node.cache_pin_source = child.cache_pin_source
        new_node.cache_pin_mode = child.cache_pin_mode
        new_node.cache_hint_motif_id = child.cache_hint_motif_id
        new_node.cache_hint_stage_id = child.cache_hint_stage_id
        new_node.cache_pin_forced_unpin = child.cache_pin_forced_unpin
        new_node.cache_pin_released_after_reuse = child.cache_pin_released_after_reuse
        new_node.cache_pin_leases = {
            key: dict(lease) for key, lease in child.cache_pin_leases.items()
        }
        self.pin_lease_created_count += len(new_node.cache_pin_leases)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _apply_priority_hint(
        self,
        node: TreeNode,
        priority: float,
        cache_pin_expires_at: Optional[float],
        cache_hint_prefix_key: Optional[str],
        cache_pin_source: str = "request",
        cache_pin_metadata: Optional[dict[str, Any]] = None,
    ):
        priority = max(0.0, float(priority or 0.0))
        if priority <= 0.0:
            return
        metadata = cache_pin_metadata or {}
        lease_key = str(metadata.get("lease_key") or "")
        consumer_key = str(metadata.get("release_consumer_key") or "")
        release_after_hits = max(0, int(metadata.get("release_after_hits") or 0))
        if lease_key and consumer_key and release_after_hits > 0:
            existing_active = node.effective_priority() > 0.0
            if lease_key in node.cache_pin_leases:
                self.pin_lease_refreshed_count += 1
            else:
                self.pin_lease_created_count += 1
            self.pin_unique_lease_created_keys.add(lease_key)
            if lease_key not in self.pin_unique_lease_expires_at:
                self.pin_unique_lease_expires_at[lease_key] = cache_pin_expires_at
            else:
                previous_expiry = self.pin_unique_lease_expires_at[lease_key]
                self.pin_unique_lease_expires_at[lease_key] = (
                    None
                    if previous_expiry is None or cache_pin_expires_at is None
                    else max(previous_expiry, cache_pin_expires_at)
                )
            node.cache_pin_leases[lease_key] = {
                "priority": priority,
                "expires_at": cache_pin_expires_at,
                "consumer_key": consumer_key,
                "remaining_hits": release_after_hits,
                "source": cache_pin_source,
            }
            if not existing_active and node.effective_priority() > 0.0:
                node_tokens = self._node_value_len(node)
                self.priority_protected_blocks += node_tokens
                self.priority_protected_blocks_by_source[
                    cache_pin_source
                ] += node_tokens
                self.protected_blocks_by_agent_type[
                    self._agent_bucket_from_metadata(cache_pin_metadata)
                ] += node_tokens
                self.pinned_blocks_total += node_tokens
            node.cache_pin_utility = max(node.cache_pin_utility, priority)
            node.cache_hint_prefix_key = cache_hint_prefix_key
            node.cache_pin_source = cache_pin_source
            node.cache_pin_released_after_reuse = False
            self._apply_cache_pin_metadata(node, cache_pin_metadata)
            return
        existing_active = node.effective_priority() > 0.0
        if not existing_active or priority >= node.effective_priority():
            if node.effective_priority() <= 0.0:
                node_tokens = self._node_value_len(node)
                self.priority_protected_blocks += node_tokens
                self.priority_protected_blocks_by_source[cache_pin_source] += node_tokens
                self.protected_blocks_by_agent_type[self._agent_bucket_from_metadata(cache_pin_metadata)] += node_tokens
                self.pinned_blocks_total += node_tokens
            node.priority = priority
            node.cache_pin_utility = priority
            node.cache_pin_expires_at = cache_pin_expires_at
            node.cache_hint_prefix_key = cache_hint_prefix_key
            node.cache_pin_source = cache_pin_source
            self._apply_cache_pin_metadata(node, cache_pin_metadata)

    @staticmethod
    def _apply_cache_pin_metadata(
        node: TreeNode, cache_pin_metadata: Optional[dict[str, Any]]
    ) -> None:
        if not isinstance(cache_pin_metadata, dict):
            return
        pin_mode = cache_pin_metadata.get("pin_mode")
        motif_id = cache_pin_metadata.get("motif_id")
        stage_id = cache_pin_metadata.get("stage_id")
        agent_type = cache_pin_metadata.get("agent_type")
        if isinstance(pin_mode, str) and pin_mode:
            node.cache_pin_mode = pin_mode
        if isinstance(motif_id, str) and motif_id:
            node.cache_hint_motif_id = motif_id
        if isinstance(stage_id, str) and stage_id:
            node.cache_hint_stage_id = stage_id
        if isinstance(agent_type, str) and agent_type:
            node.cache_hint_agent_type = agent_type
        for key, attr in (
            ("saved_prefill_ms", "cache_pin_saved_prefill_ms"),
            ("expected_queue_saving_ms", "cache_pin_expected_queue_saving_ms"),
            ("structure_release_gain_ms", "cache_pin_structure_release_gain_ms"),
            ("utility", "cache_pin_utility"),
        ):
            value = cache_pin_metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(node, attr, max(0.0, float(value)))
        if bool(cache_pin_metadata.get("pin_dynamic")):
            node.cache_pin_dynamic = True

    @staticmethod
    def _agent_bucket_from_metadata(cache_pin_metadata: Optional[dict[str, Any]]) -> str:
        if not isinstance(cache_pin_metadata, dict):
            return "unknown"
        value = str(cache_pin_metadata.get("agent_type") or "").lower()
        motif_id = str(cache_pin_metadata.get("motif_id") or "").lower()
        stage_id = str(cache_pin_metadata.get("stage_id") or "").lower()
        call_type = str(cache_pin_metadata.get("call_type") or "").lower()
        joined = " ".join((value, motif_id, stage_id, call_type))
        if "react" in joined:
            return "react"
        if motif_id and motif_id not in {"unknown", "none", "react"}:
            return "motif"
        if value in {"motif", "appworld_motif", "structured_motif"}:
            return "motif"
        return "unknown"

    @staticmethod
    def _agent_bucket(node: TreeNode) -> str:
        value = str(getattr(node, "cache_hint_agent_type", None) or "").lower()
        motif_id = str(getattr(node, "cache_hint_motif_id", None) or "").lower()
        stage_id = str(getattr(node, "cache_hint_stage_id", None) or "").lower()
        joined = " ".join((value, motif_id, stage_id))
        if "react" in joined:
            return "react"
        if motif_id and motif_id not in {"unknown", "none", "react"}:
            return "motif"
        if value in {"motif", "appworld_motif", "structured_motif"}:
            return "motif"
        return "unknown"

    def _normalize_cache_pin_ranges(
        self,
        ranges: list[dict[str, Any]],
        key_len: int,
        fallback_expires_at: Optional[float],
    ) -> list[dict[str, Any]]:
        now = time.monotonic()
        cleaned = []
        for item in ranges:
            try:
                start = int(item.get("start", 0))
                end = int(item.get("end", 0))
                priority = max(0.0, float(item.get("priority", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
            start = max(0, min(key_len, start))
            end = max(0, min(key_len, end))
            if self.page_size != 1:
                start = start // self.page_size * self.page_size
                end = end // self.page_size * self.page_size
            if end <= start or priority <= 0.0:
                continue
            expires_at = fallback_expires_at
            ttl_ms = item.get("cache_pin_ttl_ms")
            if ttl_ms is not None:
                try:
                    ttl_s = max(0.0, float(ttl_ms) / 1000.0)
                except (TypeError, ValueError):
                    ttl_s = 0.0
                expires_at = now + ttl_s if ttl_s > 0.0 else None
            if expires_at is not None and expires_at <= now:
                continue
            cleaned.append(
                {
                    "start": start,
                    "end": end,
                    "priority": priority,
                    "expires_at": expires_at,
                    "source": str(item.get("source") or "range"),
                    "pin_dynamic": bool(item.get("pin_dynamic")),
                    "saved_prefill_ms": float(item.get("saved_prefill_ms") or 0.0),
                    "expected_queue_saving_ms": float(item.get("expected_queue_saving_ms") or 0.0),
                    "structure_release_gain_ms": float(item.get("structure_release_gain_ms") or 0.0),
                    "release_after_hits": max(
                        0, int(item.get("release_after_hits") or 0)
                    ),
                    "release_consumer_key": str(item.get("release_consumer_key") or ""),
                    "lease_key": str(item.get("lease_key") or ""),
                }
            )
        return cleaned

    def _split_key_at_boundary(self, key: RadixKey, boundary: int) -> None:
        if boundary <= 0 or boundary >= len(key):
            return
        self._match_prefix_helper(self.root_node, key[:boundary])

    def _iter_nodes_for_key(self, key: RadixKey):
        node = self.root_node
        remaining = key
        offset = 0
        while len(remaining) > 0:
            child_key = self.get_child_key_fn(remaining)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = self.key_match_fn(child.key, remaining)
            if prefix_len <= 0:
                break
            yield child, offset, offset + prefix_len
            node = child
            remaining = remaining[prefix_len:]
            offset += prefix_len

    def _apply_priority_ranges(
        self,
        key: RadixKey,
        ranges: list[dict[str, Any]],
        fallback_expires_at: Optional[float],
        cache_hint_prefix_key: Optional[str],
        cache_pin_metadata: Optional[dict[str, Any]],
    ) -> None:
        cleaned = self._normalize_cache_pin_ranges(
            ranges, len(key), fallback_expires_at
        )
        if not cleaned:
            return
        boundaries = sorted(
            {
                boundary
                for item in cleaned
                for boundary in (item["start"], item["end"])
                if 0 < boundary < len(key)
            }
        )
        for boundary in boundaries:
            self._split_key_at_boundary(key, boundary)

        for node, start, end in self._iter_nodes_for_key(key):
            best = None
            for item in cleaned:
                if item["start"] <= start and end <= item["end"]:
                    if best is None or item["priority"] > best["priority"]:
                        best = item
            if best is None:
                continue
            self._apply_priority_hint(
                node,
                best["priority"],
                best["expires_at"],
                cache_hint_prefix_key,
                cache_pin_source=best["source"],
                cache_pin_metadata={
                    **(cache_pin_metadata or {}),
                    "pin_dynamic": best.get("pin_dynamic", False),
                    "saved_prefill_ms": best.get("saved_prefill_ms", 0.0),
                    "expected_queue_saving_ms": best.get("expected_queue_saving_ms", 0.0),
                    "structure_release_gain_ms": best.get("structure_release_gain_ms", 0.0),
                    "release_after_hits": best.get("release_after_hits", 0),
                    "release_consumer_key": best.get("release_consumer_key", ""),
                    "lease_key": best.get("lease_key", ""),
                    "utility": best["priority"],
                },
            )

    @staticmethod
    def _node_value_len(node: TreeNode) -> int:
        value = getattr(node, "value", None)
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: float = 0.0,
        cache_pin_expires_at: Optional[float] = None,
        cache_hint_prefix_key: Optional[str] = None,
        cache_pin_ranges: Optional[list[dict[str, Any]]] = None,
        cache_pin_metadata: Optional[dict[str, Any]] = None,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0.0
        original_key = key
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path so shared ancestors are protected too.
        self._apply_priority_hint(
            node,
            priority,
            cache_pin_expires_at,
            cache_hint_prefix_key,
            cache_pin_metadata=cache_pin_metadata,
        )
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                self._apply_priority_hint(
                    new_node,
                    priority,
                    cache_pin_expires_at,
                    cache_hint_prefix_key,
                    cache_pin_metadata=cache_pin_metadata,
                )
                node = new_node
            else:
                self._apply_priority_hint(
                    node,
                    priority,
                    cache_pin_expires_at,
                    cache_hint_prefix_key,
                    cache_pin_metadata=cache_pin_metadata,
                )

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.cache_pin_expires_at = cache_pin_expires_at
            new_node.cache_hint_prefix_key = cache_hint_prefix_key
            new_node.cache_pin_source = "request" if priority > 0.0 else None
            self._apply_cache_pin_metadata(new_node, cache_pin_metadata)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            if new_node.effective_priority() > 0.0:
                node_tokens = self._node_value_len(new_node)
                self.priority_protected_blocks += node_tokens
                self.priority_protected_blocks_by_source["request"] += node_tokens
                self.protected_blocks_by_agent_type[
                    self._agent_bucket_from_metadata(cache_pin_metadata)
                ] += node_tokens
                self.pinned_blocks_total += node_tokens
                self.actual_prefill_tokens += node_tokens
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        if cache_pin_ranges:
            self._apply_priority_ranges(
                original_key,
                cache_pin_ranges,
                fallback_expires_at=cache_pin_expires_at,
                cache_hint_prefix_key=cache_hint_prefix_key,
                cache_pin_metadata=cache_pin_metadata,
            )
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
