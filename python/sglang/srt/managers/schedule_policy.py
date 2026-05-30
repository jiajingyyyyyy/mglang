from __future__ import annotations

import logging
import math
import os
import time

from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.utils import get_bool_env_var

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
STRUCTURED_HINT_GROUP_CAP = int(os.environ.get("SGLANG_STRUCTURED_HINT_GROUP_CAP", "4"))
STRUCTURED_HINT_POLICY_DEBUG_LOG = get_bool_env_var(
    "SGLANG_STRUCTURED_HINT_POLICY_DEBUG_LOG"
)
SLO_PREFIX_DEFAULT_BUDGET_MS = float(
    os.environ.get("SGLANG_SLO_PREFIX_DEFAULT_BUDGET_MS", "10000")
)
SLO_PREFIX_REACT_BUDGET_MS = float(
    os.environ.get("SGLANG_SLO_PREFIX_REACT_BUDGET_MS", str(SLO_PREFIX_DEFAULT_BUDGET_MS))
)
SLO_PREFIX_MOTIF_BUDGET_MS = float(
    os.environ.get("SGLANG_SLO_PREFIX_MOTIF_BUDGET_MS", str(SLO_PREFIX_DEFAULT_BUDGET_MS))
)
SLO_PREFIX_EPS_MS = float(os.environ.get("SGLANG_SLO_PREFIX_EPS_MS", "50"))
SLO_PREFIX_SLACK_GAMMA = float(os.environ.get("SGLANG_SLO_PREFIX_SLACK_GAMMA", "1.0"))
SLO_PREFIX_MARGINAL_COST_GAMMA = float(
    os.environ.get("SGLANG_SLO_PREFIX_MARGINAL_COST_GAMMA", "0.25")
)
SLO_BOOST_REACT_ALPHA = float(os.environ.get("SGLANG_SLO_BOOST_REACT_ALPHA", "1.0"))
SLO_BOOST_MOTIF_BETA = float(os.environ.get("SGLANG_SLO_BOOST_MOTIF_BETA", "2.0"))
SLO_BOOST_REACT_TARGET_MS = float(
    os.environ.get("SGLANG_SLO_BOOST_REACT_TARGET_MS", "30000")
)
SLO_BOOST_MOTIF_LAG_TARGET_MS = float(
    os.environ.get("SGLANG_SLO_BOOST_MOTIF_LAG_TARGET_MS", "10000")
)
SLO_BOOST_MAX_MULTIPLIER = float(
    os.environ.get("SGLANG_SLO_BOOST_MAX_MULTIPLIER", "4.0")
)
SLO_PREFIX_MIX_DFS_SLOTS = int(os.environ.get("SGLANG_SLO_PREFIX_MIX_DFS_SLOTS", "3"))
SLO_PREFIX_MIX_SLO_SLOTS = int(os.environ.get("SGLANG_SLO_PREFIX_MIX_SLO_SLOTS", "1"))
SLO_COST_PREFIX_DECODE_WEIGHT = float(
    os.environ.get("SGLANG_SLO_COST_PREFIX_DECODE_WEIGHT", "0.0")
)
SLO_COST_PREFIX_REACT_TAU_S = float(
    os.environ.get("SGLANG_SLO_COST_PREFIX_REACT_TAU_S", "5.0")
)
SLO_COST_PREFIX_MOTIF_TAU_S = float(
    os.environ.get("SGLANG_SLO_COST_PREFIX_MOTIF_TAU_S", "10.0")
)
SLO_COST_PREFIX_SEMANTIC_WEIGHT = float(
    os.environ.get("SGLANG_SLO_COST_PREFIX_SEMANTIC_WEIGHT", "1.0")
)
SLO_COST_PREFIX_MIN_REUSE_VALUE = float(
    os.environ.get("SGLANG_SLO_COST_PREFIX_MIN_REUSE_VALUE", "1.0")
)
SLO_COST_PREFIX_SEMANTIC_GROUP = os.environ.get(
    "SGLANG_SLO_COST_PREFIX_SEMANTIC_GROUP", "stage"
).strip().lower()
MPLS_DEFAULT_BUDGET_MS = float(os.environ.get("SGLANG_MPLS_DEFAULT_BUDGET_MS", "10000"))
logger = logging.getLogger(__name__)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_in_seq_split
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    SLO_BOOSTED_DFS = "slo-boosted-dfs"  # DFS weighted by online React/Motif SLO boost
    SLO_PREFIX_DFS = "slo-prefix-dfs"  # prefix reuse weighted by SLO slack
    SLO_MARGINAL_PREFIX_DFS = "slo-marginal-prefix-dfs"  # SLO-prefix DFS with weak marginal prefill cost
    SLO_COST_PREFIX_DFS = "slo-cost-prefix-dfs"  # prefix reuse weighted by SLO urgency/cost
    SEMANTIC_SLO_COST_DFS = "semantic-slo-cost-dfs"  # motif semantic overlay over SLO/cost prefix DFS
    SLO_PREFIX_MIXED_DFS = "slo-prefix-mixed-dfs"  # bounded SLO slice over DFS
    MPLS = "mpls"  # deadline-guarded motif prefix lease scheduling


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    EDF_LIKE = "edf-like"  # earliest remaining SLO slack first
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch
    STRUCTURED_HINT = "structured-hint"  # locality-aware advisory request hints


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        self.priority_sign = 1 if schedule_low_priority_values_first else -1

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()
        self.slo_prefix_mixed_slo_order_rids = []
        self.mpls_stats = defaultdict(float)

    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> bool:
        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return False

        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            elif policy == CacheAwarePolicy.SLO_BOOSTED_DFS:
                SchedulePolicy._sort_by_slo_boosted_dfs(
                    waiting_queue, self.tree_cache
                )
            elif policy == CacheAwarePolicy.SLO_PREFIX_DFS:
                SchedulePolicy._sort_by_slo_prefix_dfs(
                    waiting_queue, self.tree_cache
                )
            elif policy == CacheAwarePolicy.SLO_MARGINAL_PREFIX_DFS:
                SchedulePolicy._sort_by_slo_marginal_prefix_dfs(
                    waiting_queue, self.tree_cache
                )
            elif policy == CacheAwarePolicy.SLO_COST_PREFIX_DFS:
                SchedulePolicy._sort_by_slo_cost_prefix_dfs(
                    waiting_queue, self.tree_cache, semantic_overlay=False
                )
            elif policy == CacheAwarePolicy.SEMANTIC_SLO_COST_DFS:
                SchedulePolicy._sort_by_slo_cost_prefix_dfs(
                    waiting_queue, self.tree_cache, semantic_overlay=True
                )
            elif policy == CacheAwarePolicy.SLO_PREFIX_MIXED_DFS:
                self._sort_by_slo_prefix_mixed_dfs(waiting_queue, self.tree_cache)
            elif policy == CacheAwarePolicy.MPLS:
                SchedulePolicy._sort_by_mpls(
                    waiting_queue,
                    temporary_deprioritized,
                    self.mpls_stats,
                )
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.EDF_LIKE:
                SchedulePolicy._sort_by_edf_like(waiting_queue)
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            elif policy == CacheAgnosticPolicy.STRUCTURED_HINT:
                SchedulePolicy._sort_by_structured_hints(
                    waiting_queue,
                    running_batch,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")
        return prefix_computed

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            # NOTE: the prefix_indices must always be aligned with last_node
            match_result = self.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                )
            )
            (
                r.prefix_indices,
                r.last_node,
                r.last_host_node,
                r.host_hit_length,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                    )
                )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_indices)
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_mpls(
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        mpls_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Sort by deadline-guarded prefix leases without external waiting.

        MPLS is intentionally work-conserving here: it only reorders the
        backend-visible waiting queue.  The selected lease is moved to the
        front, and the normal SGLang prefill adder may continue filling the
        batch from following requests if capacity remains.
        """
        if len(waiting_queue) <= 1:
            return

        now = time.perf_counter()
        if mpls_stats is not None:
            mpls_stats["calls"] += 1
            mpls_stats["waiting_req_count_total"] += len(waiting_queue)
        ready_prefix_index = SchedulePolicy._mpls_ready_prefix_index(waiting_queue)
        leases: Dict[tuple, dict] = {}
        for original_index, req in enumerate(waiting_queue):
            for entry in SchedulePolicy._mpls_prefix_ladder(req):
                key = SchedulePolicy._mpls_lease_key(req, entry)
                lease = leases.setdefault(
                    key,
                    {
                        "key": key,
                        "level": entry["level"],
                        "prefix_hash": entry["prefix_hash"],
                        "prefix_len": entry["prefix_len"],
                        "reqs": [],
                        "first_arrival": float("inf"),
                        "deadline": float("inf"),
                        "prefix_gain_ms": 0.0,
                        "release_gain_ms": 0.0,
                        "bridge_release_gain_ms": 0.0,
                        "total_gain_ms": 0.0,
                    },
                )
                lease["reqs"].append((original_index, req))
                arrival = SchedulePolicy._mpls_arrival_s(req)
                lease["first_arrival"] = min(lease["first_arrival"], arrival)
                lease["deadline"] = min(
                    lease["deadline"], SchedulePolicy._mpls_deadline_s(req, now)
                )

        if not leases:
            if mpls_stats is not None:
                mpls_stats["fallback_longest_prefix_count"] += 1
            SchedulePolicy._sort_by_longest_prefix(
                waiting_queue, temporary_deprioritized
            )
            return

        scored_leases = []
        for lease in leases.values():
            req_count = len(lease["reqs"])
            if req_count <= 0:
                continue
            denominator = max(1e-3, lease["deadline"] - lease["first_arrival"])
            normalized_lag = (now - lease["first_arrival"]) / denominator
            prefix_gain = SchedulePolicy._mpls_prefix_gain_ms(lease)
            direct_release_gain = sum(
                SchedulePolicy._mpls_downstream_release_gain_ms(req)
                for _index, req in lease["reqs"]
            )
            bridge_release_gain = sum(
                SchedulePolicy._mpls_bridge_release_gain_ms(req, ready_prefix_index)
                for _index, req in lease["reqs"]
            )
            release_gain = direct_release_gain + bridge_release_gain
            total_gain = prefix_gain + release_gain
            lease["normalized_lag"] = normalized_lag
            lease["prefix_gain"] = prefix_gain
            lease["prefix_gain_ms"] = prefix_gain
            lease["release_gain_ms"] = release_gain
            lease["bridge_release_gain_ms"] = bridge_release_gain
            lease["total_gain_ms"] = total_gain
            scored_leases.append(lease)

        if not scored_leases:
            if mpls_stats is not None:
                mpls_stats["fallback_longest_prefix_count"] += 1
            SchedulePolicy._sort_by_longest_prefix(
                waiting_queue, temporary_deprioritized
            )
            return

        expired = [lease for lease in scored_leases if lease["deadline"] <= now]
        if mpls_stats is not None:
            mpls_stats["active_lease_count_total"] += len(scored_leases)
            mpls_stats["expired_lease_count_total"] += len(expired)
        if expired:
            selected = min(
                expired,
                key=lambda lease: (
                    lease["deadline"],
                    lease["first_arrival"],
                    -lease["total_gain_ms"],
                    -lease["prefix_gain_ms"],
                ),
            )
            selected_expired = True
        else:
            selected = max(
                scored_leases,
                key=lambda lease: (
                    lease["total_gain_ms"],
                    lease["release_gain_ms"],
                    lease["prefix_gain_ms"],
                    lease["prefix_len"],
                    -lease["first_arrival"],
                ),
            )
            selected_expired = False
            if selected["total_gain_ms"] <= 0:
                if mpls_stats is not None:
                    mpls_stats["fallback_longest_prefix_count"] += 1
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
                return

        selected_ids = {id(req) for _index, req in selected["reqs"]}
        selected_reqs = [req for _index, req in selected["reqs"]]
        if selected_expired:
            selected_reqs.sort(
                key=lambda req: (
                    SchedulePolicy._mpls_deadline_s(req, now),
                    SchedulePolicy._mpls_arrival_s(req),
                    str(req.rid),
                )
            )
        else:
            selected_reqs.sort(
                key=lambda req: (
                    SchedulePolicy._mpls_arrival_s(req),
                    str(req.rid),
                )
            )
        rest = [req for req in waiting_queue if id(req) not in selected_ids]
        SchedulePolicy._sort_by_longest_prefix(rest, temporary_deprioritized)
        waiting_queue[:] = selected_reqs + rest
        if mpls_stats is not None:
            SchedulePolicy._record_mpls_selection(
                mpls_stats, selected, selected_reqs, selected_expired
            )

    @staticmethod
    def _record_mpls_selection(
        mpls_stats: Dict[str, float],
        selected: dict,
        selected_reqs: List[Req],
        selected_expired: bool,
    ) -> None:
        mpls_stats["selected_count"] += 1
        mpls_stats["selected_expired_count"] += 1 if selected_expired else 0
        mpls_stats["selected_req_count_total"] += len(selected_reqs)
        mpls_stats["selected_prefix_len_total"] += float(selected["prefix_len"])
        mpls_stats["selected_prefix_gain_total"] += float(selected["prefix_gain"])
        mpls_stats["selected_release_gain_ms_total"] += float(
            selected.get("release_gain_ms") or 0.0
        )
        mpls_stats["selected_bridge_release_gain_ms_total"] += float(
            selected.get("bridge_release_gain_ms") or 0.0
        )
        mpls_stats["selected_total_gain_ms_total"] += float(
            selected.get("total_gain_ms") or 0.0
        )
        mpls_stats["selected_normalized_lag_total"] += float(
            selected["normalized_lag"]
        )

    def get_mpls_stats(self) -> dict:
        selected_count = max(1.0, float(self.mpls_stats.get("selected_count") or 0.0))
        calls = max(1.0, float(self.mpls_stats.get("calls") or 0.0))
        stats = dict(self.mpls_stats)
        stats["selected_prefix_len_avg"] = (
            float(stats.get("selected_prefix_len_total") or 0.0) / selected_count
        )
        stats["selected_prefix_gain_avg"] = (
            float(stats.get("selected_prefix_gain_total") or 0.0) / selected_count
        )
        stats["selected_release_gain_ms_avg"] = (
            float(stats.get("selected_release_gain_ms_total") or 0.0)
            / selected_count
        )
        stats["selected_bridge_release_gain_ms_avg"] = (
            float(stats.get("selected_bridge_release_gain_ms_total") or 0.0)
            / selected_count
        )
        stats["selected_total_gain_ms_avg"] = (
            float(stats.get("selected_total_gain_ms_total") or 0.0)
            / selected_count
        )
        stats["selected_normalized_lag_avg"] = (
            float(stats.get("selected_normalized_lag_total") or 0.0) / selected_count
        )
        stats["selected_req_count_avg"] = (
            float(stats.get("selected_req_count_total") or 0.0) / selected_count
        )
        stats["active_lease_count_avg"] = (
            float(stats.get("active_lease_count_total") or 0.0) / calls
        )
        stats["expired_lease_count_avg"] = (
            float(stats.get("expired_lease_count_total") or 0.0) / calls
        )
        return stats

    @staticmethod
    def _mpls_prefix_ladder(req: Req) -> List[dict]:
        hint = getattr(req, "structured_hints", None)
        raw_ladder = getattr(hint, "prefix_ladder", None) if hint is not None else None
        entries: List[dict] = []
        if isinstance(raw_ladder, list):
            for raw in raw_ladder:
                if not isinstance(raw, dict):
                    continue
                prefix_hash = raw.get("prefix_hash") or raw.get("hash")
                prefix_len = SchedulePolicy._safe_int(raw.get("prefix_len") or raw.get("len"), 0)
                if isinstance(prefix_hash, str) and prefix_hash and prefix_len > 0:
                    entries.append(
                        {
                            "level": str(raw.get("level") or "prefix"),
                            "prefix_hash": prefix_hash,
                            "prefix_len": prefix_len,
                        }
                    )
            return entries

        prefix_hash = (
            getattr(hint, "prompt_prefix_hash", None)
            or getattr(hint, "prefix_key", None)
            if hint is not None
            else None
        )
        prefix_len = SchedulePolicy._safe_int(
            getattr(hint, "static_prefix_len", None) if hint is not None else None,
            0,
        )
        if isinstance(prefix_hash, str) and prefix_hash and prefix_len > 0:
            return [
                {
                    "level": "hint_prefix",
                    "prefix_hash": prefix_hash,
                    "prefix_len": prefix_len,
                }
            ]
        return []

    @staticmethod
    def _mpls_ready_prefix_index(waiting_queue: List[Req]) -> Dict[tuple, dict]:
        index: Dict[tuple, dict] = {}
        for req in waiting_queue:
            for entry in SchedulePolicy._mpls_prefix_ladder(req):
                key = (entry["level"], entry["prefix_hash"])
                bucket = index.setdefault(
                    key,
                    {"req_ids": set(), "prefix_len": 0.0, "ms_per_token": 0.0},
                )
                bucket["req_ids"].add(id(req))
                bucket["prefix_len"] = max(
                    float(bucket["prefix_len"]), float(entry["prefix_len"])
                )
                bucket["ms_per_token"] = max(
                    float(bucket["ms_per_token"]),
                    SchedulePolicy._mpls_prefill_ms_per_token(req),
                )
        return index

    @staticmethod
    def _mpls_lease_key(req: Req, entry: dict) -> tuple:
        hint = getattr(req, "structured_hints", None)
        return (
            getattr(req, "extra_key", None) or "",
            getattr(req, "lora_id", None) or "",
            getattr(req, "grammar_key", None) or "",
            getattr(hint, "grammar_id", None) if hint is not None else "",
            getattr(hint, "max_tokens_bucket", None) if hint is not None else "",
            entry["level"],
            entry["prefix_hash"],
            int(entry["prefix_len"]),
        )

    @staticmethod
    def _mpls_arrival_s(req: Req) -> float:
        return float(
            getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0)
            or 0.0
        )

    @staticmethod
    def _mpls_prefill_ms_per_token(req: Req) -> float:
        hint = getattr(req, "structured_hints", None)
        saved_ms = getattr(hint, "saved_prefill_cost_ms", None) if hint else None
        static_len = getattr(hint, "static_prefix_len", None) if hint else None
        if (
            isinstance(saved_ms, (int, float))
            and not isinstance(saved_ms, bool)
            and isinstance(static_len, (int, float))
            and not isinstance(static_len, bool)
            and float(saved_ms) > 0
            and float(static_len) > 0
        ):
            return max(1e-6, float(saved_ms) / float(static_len))
        return 1.0

    @staticmethod
    def _mpls_prefix_gain_ms(lease: dict) -> float:
        reqs = [req for _index, req in lease["reqs"]]
        if len(reqs) <= 1:
            return 0.0
        ms_per_token = max(
            SchedulePolicy._mpls_prefill_ms_per_token(req) for req in reqs
        )
        return (
            max(0.0, float(len(reqs) - 1))
            * max(0.0, float(lease["prefix_len"]))
            * ms_per_token
        )

    @staticmethod
    def _mpls_downstream_release_gain_ms(req: Req) -> float:
        hint = getattr(req, "structured_hints", None)
        raw = getattr(hint, "downstream_release_credit", None) if hint else None
        if not raw:
            return 0.0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(0.0, float(raw))
        if not isinstance(raw, dict):
            return 0.0

        for key in ("release_gain_ms", "saved_ms", "score_ms"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0.0, float(value))

        for key in ("score", "credit", "release_gain"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return (
                    max(0.0, float(value))
                    * SchedulePolicy._mpls_prefill_ms_per_token(req)
                    * SchedulePolicy._mpls_structural_unlock_multiplier(raw)
                )

        expected_ready = raw.get("expected_ready_count")
        prefix_len = raw.get("downstream_prefix_len", raw.get("prefix_len"))
        if (
            isinstance(expected_ready, (int, float))
            and not isinstance(expected_ready, bool)
            and isinstance(prefix_len, (int, float))
            and not isinstance(prefix_len, bool)
        ):
            return (
                    max(0.0, float(expected_ready))
                    * max(0.0, float(prefix_len))
                    * SchedulePolicy._mpls_prefill_ms_per_token(req)
                    * SchedulePolicy._mpls_explicit_structural_unlock_multiplier(raw)
                )
        structural_score = SchedulePolicy._mpls_structural_unlock_multiplier(raw)
        return max(0.0, structural_score - 1.0) * SchedulePolicy._mpls_prefill_ms_per_token(req)

    @staticmethod
    def _mpls_explicit_structural_unlock_multiplier(raw: dict) -> float:
        if not isinstance(raw, dict):
            return 1.0
        value = raw.get("structural_unlock_score")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            value = raw.get("critical_path_unlock_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.25, min(4.0, float(value)))
        return 1.0

    @staticmethod
    def _mpls_structural_unlock_multiplier(raw: dict) -> float:
        if not isinstance(raw, dict):
            return 1.0
        value = raw.get("structural_unlock_score")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            value = raw.get("critical_path_unlock_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.25, min(4.0, float(value)))
        downstream = SchedulePolicy._safe_float(raw.get("downstream_reachable_count"), 0.0)
        depth = SchedulePolicy._safe_float(raw.get("critical_path_depth"), 0.0)
        if downstream <= 0.0 and depth <= 0.0:
            return 1.0
        return max(0.25, min(4.0, 1.0 + 0.25 * downstream + 0.15 * depth))

    @staticmethod
    def _mpls_bridge_release_gain_ms(req: Req, ready_prefix_index: Dict[tuple, dict]) -> float:
        hint = getattr(req, "structured_hints", None)
        raw = getattr(hint, "downstream_release_credit", None) if hint else None
        if not isinstance(raw, dict) or not ready_prefix_index:
            return 0.0
        future_entries = SchedulePolicy._mpls_future_prefix_entries(raw)
        if not future_entries:
            return 0.0
        expected_ready = SchedulePolicy._safe_float(raw.get("expected_ready_count"), 1.0)
        expected_ready = max(1.0, expected_ready)
        confidence = SchedulePolicy._safe_float(raw.get("confidence"), 1.0)
        confidence = max(0.0, min(1.0, confidence))
        ms_per_token = SchedulePolicy._mpls_prefill_ms_per_token(req)

        best_gain = 0.0
        for entry in future_entries:
            bucket = ready_prefix_index.get((entry["level"], entry["prefix_hash"]))
            if not bucket:
                continue
            req_ids = bucket.get("req_ids")
            ready_match_count = (
                max(0, len(req_ids - {id(req)})) if isinstance(req_ids, set) else 1
            )
            if ready_match_count <= 0:
                continue
            shared_len = min(
                float(entry["prefix_len"]),
                float(bucket.get("prefix_len") or 0.0),
            )
            if shared_len <= 0.0:
                continue
            best_gain = max(
                best_gain,
                shared_len
                * max(ms_per_token, float(bucket.get("ms_per_token") or 0.0))
                * expected_ready
                * float(ready_match_count)
                * confidence
                * SchedulePolicy._mpls_explicit_structural_unlock_multiplier(raw),
            )
        return max(0.0, best_gain)

    @staticmethod
    def _mpls_future_prefix_entries(raw: dict) -> List[dict]:
        entries: List[dict] = []
        rows = raw.get("downstream_prefix_ladder") or raw.get("future_prefix_ladder")
        if isinstance(rows, list):
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                prefix_hash = row.get("prefix_hash") or row.get("hash")
                prefix_len = SchedulePolicy._safe_int(
                    row.get("prefix_len") or row.get("len"), 0
                )
                if isinstance(prefix_hash, str) and prefix_hash and prefix_len > 0:
                    entries.append(
                        {
                            "level": str(row.get("level") or f"future_{index}"),
                            "prefix_hash": prefix_hash,
                            "prefix_len": prefix_len,
                        }
                    )
        prefix_hash = raw.get("downstream_prefix_hash") or raw.get("prefix_hash")
        prefix_len = SchedulePolicy._safe_int(
            raw.get("downstream_prefix_len") or raw.get("prefix_len"), 0
        )
        if isinstance(prefix_hash, str) and prefix_hash and prefix_len > 0:
            entries.append(
                {
                    "level": str(
                        raw.get("downstream_prefix_level")
                        or raw.get("level")
                        or "prefix"
                    ),
                    "prefix_hash": prefix_hash,
                    "prefix_len": prefix_len,
                }
            )
        return entries

    @staticmethod
    def _mpls_deadline_s(req: Req, now: float) -> float:
        hint = getattr(req, "structured_hints", None)
        wait_entry = SchedulePolicy._mpls_arrival_s(req)
        latest_start_ms = None
        if hint is not None:
            latest_start_ms = (
                getattr(hint, "internal_latest_start_ms", None)
                or getattr(hint, "latest_start_ms", None)
            )
        if isinstance(latest_start_ms, (int, float)) and not isinstance(
            latest_start_ms, bool
        ):
            latest_start_ms = float(latest_start_ms)
            if latest_start_ms > 1_000_000_000_000:
                return latest_start_ms / 1000.0 - (time.time() - now)
            if latest_start_ms > 1_000_000_000:
                return latest_start_ms / 1000.0 - (time.time() - now)
            if wait_entry:
                # Treat small values as relative latest-start budget in ms.
                return wait_entry + max(0.0, latest_start_ms / 1000.0)

        raw_slack = SchedulePolicy._slo_prefix_raw_slack_s(req, now)
        if math.isfinite(raw_slack):
            return now + raw_slack
        return (wait_entry or now) + MPLS_DEFAULT_BUDGET_MS / 1000.0

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_slo_boosted_dfs(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sort by DFS subtree weight with a minimal online class boost.

        The base signal is still radix DFS:

            base_score(u) = prefix_len(u) * waiting_count(u)

        We only multiply the base score by a class-level boost:

            React-heavy u: base_score * (1 + alpha * react_risk)
            Motif-heavy u: base_score * (1 + beta * motif_lag)

        For mixed subtrees, the boost is interpolated by class composition.
        This keeps the mechanism simple and reuse-preserving: motif/react hints
        affect traversal priority, not cache identity.
        """

        now = time.perf_counter()
        last_node_to_reqs = defaultdict(list)
        react_waits = []
        motif_waits = []
        node_to_weight = defaultdict(int)
        node_to_react_count = defaultdict(int)
        node_to_motif_count = defaultdict(int)
        node_to_score = defaultdict(float)

        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)
            node_to_weight[req.last_node] += 1
            agent_type = SchedulePolicy._slo_boost_agent_type(req)
            wait_s = SchedulePolicy._queue_wait_s(req, now)
            if agent_type == "react":
                node_to_react_count[req.last_node] += 1
                react_waits.append(wait_s)
            elif agent_type == "motif":
                node_to_motif_count[req.last_node] += 1
                motif_waits.append(wait_s)

        react_p95 = SchedulePolicy._percentile(react_waits, 0.95)
        motif_p95 = SchedulePolicy._percentile(motif_waits, 0.95)
        react_target_s = max(0.001, SLO_BOOST_REACT_TARGET_MS / 1000.0)
        motif_lag_target_s = max(0.001, SLO_BOOST_MOTIF_LAG_TARGET_MS / 1000.0)
        react_risk = max(0.0, react_p95 / react_target_s - 1.0)
        motif_lag = max(0.0, (motif_p95 - react_p95) / motif_lag_target_s)

        SchedulePolicy._calc_slo_boosted_score(
            tree_cache.root_node,
            depth=0,
            node_to_weight=node_to_weight,
            node_to_react_count=node_to_react_count,
            node_to_motif_count=node_to_motif_count,
            node_to_score=node_to_score,
            react_risk=react_risk,
            motif_lag=motif_lag,
        )

        waiting_queue.clear()
        SchedulePolicy._get_slo_prefix_dfs_priority(
            tree_cache.root_node,
            node_to_score,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
            now,
        )

    @staticmethod
    def _sort_by_slo_prefix_dfs(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sort by prefix reuse per aggregate remaining SLO slack.

        This is a small semantic/SLO-aware overlay on top of radix DFS. Cache
        identity is unchanged: requests still match by real token prefix. The
        scheduler only changes which radix subtree is explored first:

            score(subtree) = shared_prefix_len * waiting_count / sum(slack)^gamma

        where slack is derived from request wait time and an agent-class budget.
        """
        now = time.perf_counter()
        last_node_to_reqs = defaultdict(list)
        node_to_slack_sum = defaultdict(float)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)
            node_to_slack_sum[req.last_node] += SchedulePolicy._slo_prefix_slack_s(
                req, now
            )

        node_to_weight = defaultdict(int)
        node_to_score = defaultdict(float)
        for node, reqs in last_node_to_reqs.items():
            node_to_weight[node] = len(reqs)

        SchedulePolicy._calc_slo_prefix_score(
            tree_cache.root_node,
            depth=0,
            node_to_weight=node_to_weight,
            node_to_slack_sum=node_to_slack_sum,
            node_to_score=node_to_score,
        )

        waiting_queue.clear()
        SchedulePolicy._get_slo_prefix_dfs_priority(
            tree_cache.root_node,
            node_to_score,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
            now,
        )

    @staticmethod
    def _slo_prefix_slack_s(req: Req, now: float) -> float:
        raw_slack = SchedulePolicy._slo_prefix_raw_slack_s(req, now)
        eps_s = max(0.001, SLO_PREFIX_EPS_MS / 1000.0)
        return max(eps_s, raw_slack)

    @staticmethod
    def _slo_prefix_raw_slack_s(req: Req, now: float) -> float:
        hint = getattr(req, "structured_hints", None)
        agent_type = SchedulePolicy._slo_agent_type(req)
        wait_entry = getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0)
        deadline_ms = getattr(hint, "deadline_ms", None) if hint is not None else None
        if isinstance(deadline_ms, (int, float)) and not isinstance(deadline_ms, bool):
            if deadline_ms > 1_000_000_000_000:
                # Epoch milliseconds.
                return max(0.0, deadline_ms / 1000.0 - time.time())
            if wait_entry:
                return wait_entry + max(0.0, float(deadline_ms) / 1000.0) - now

        if agent_type == "motif":
            budget_ms = SLO_PREFIX_MOTIF_BUDGET_MS
        elif agent_type == "react":
            budget_ms = SLO_PREFIX_REACT_BUDGET_MS
        else:
            budget_ms = SLO_PREFIX_DEFAULT_BUDGET_MS

        wait_s = max(0.0, now - wait_entry) if wait_entry else 0.0
        return budget_ms / 1000.0 - wait_s

    @staticmethod
    def _slo_agent_type(req: Req) -> str:
        hint = getattr(req, "structured_hints", None)
        return str(
            getattr(hint, "agent_type", None)
            or getattr(hint, "trace_label", None)
            or ""
        ).lower()

    @staticmethod
    def _slo_stage_id(req: Req) -> str:
        hint = getattr(req, "structured_hints", None)
        return str(getattr(hint, "stage_id", None) or "").lower()

    @staticmethod
    def _slo_boost_agent_type(req: Req) -> str:
        agent_type = SchedulePolicy._slo_agent_type(req)
        stage_id = SchedulePolicy._slo_stage_id(req)
        if agent_type == "motif" and stage_id == "react_fallback":
            # The boost is meant to rescue structured Motif stages. Fallback
            # prompts are ReAct-like long-prefix calls; treating them as Motif
            # amplifies the wrong subtree and hurts both React and Motif tails.
            return "react"
        return agent_type

    @staticmethod
    def _queue_wait_s(req: Req, now: float) -> float:
        wait_entry = getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0)
        return max(0.0, now - wait_entry) if wait_entry else 0.0

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(math.ceil(percentile * len(sorted_values))) - 1
        idx = min(max(idx, 0), len(sorted_values) - 1)
        return float(sorted_values[idx])

    @staticmethod
    def _slo_prefix_marginal_cost(req: Req) -> float:
        """Estimate the request's immediate uncached prefill cost.

        Full prompt length is the wrong denominator for cache-aware scheduling:
        a long request whose prefix is already cached is cheap at admission time.
        SGLang's prefill budget is driven by the uncached extension length, so
        this helper approximates that marginal cost before add_one_req recomputes
        req.extend_input_len.
        """

        fill_len = len(getattr(req, "origin_input_ids", None) or [])
        fill_len += len(getattr(req, "output_ids", None) or [])
        prefix_len = SchedulePolicy._safe_len(getattr(req, "prefix_indices", None))
        host_hit_length = max(
            0,
            SchedulePolicy._safe_int(getattr(req, "host_hit_length", 0), default=0),
        )
        return max(1.0, float(fill_len - prefix_len - host_hit_length))

    @staticmethod
    def _slo_cost_urgency(req: Req, now: float) -> float:
        hint = getattr(req, "structured_hints", None)
        agent_type = str(
            getattr(hint, "agent_type", None)
            or getattr(hint, "trace_label", None)
            or ""
        ).lower()
        tau = SLO_COST_PREFIX_MOTIF_TAU_S if agent_type == "motif" else SLO_COST_PREFIX_REACT_TAU_S
        tau = max(0.001, float(tau))
        raw_slack = SchedulePolicy._slo_prefix_raw_slack_s(req, now)
        exponent = max(-30.0, min(30.0, -raw_slack / tau))
        return math.exp(exponent)

    @staticmethod
    def _slo_cost_service_cost(req: Req) -> float:
        prompt_tokens = len(getattr(req, "origin_input_ids", None) or [])
        prompt_tokens += len(getattr(req, "output_ids", None) or [])
        max_new_tokens = getattr(getattr(req, "sampling_params", None), "max_new_tokens", 0)
        try:
            output_budget = min(max(0, int(max_new_tokens or 0)), CLIP_MAX_NEW_TOKENS)
        except (TypeError, ValueError):
            output_budget = 0
        return max(
            1.0,
            float(prompt_tokens) + max(0.0, SLO_COST_PREFIX_DECODE_WEIGHT) * float(output_budget),
        )

    @staticmethod
    def _semantic_fragment_key(req: Req) -> Optional[tuple]:
        hint = getattr(req, "structured_hints", None)
        if hint is None:
            return None
        agent_type = str(
            getattr(hint, "agent_type", None)
            or getattr(hint, "trace_label", None)
            or ""
        ).lower()
        if agent_type != "motif":
            # ReAct already appears as one large token-prefix subtree in the
            # workloads we care about. The semantic overlay is deliberately
            # Motif-only so it captures fragmentation rather than amplifying
            # the dominant ReAct prefix.
            return None

        motif_id = str(getattr(hint, "motif_id", None) or "unknown_motif")
        stage_id = str(getattr(hint, "stage_id", None) or "unknown_stage")
        if SLO_COST_PREFIX_SEMANTIC_GROUP == "motif":
            return ("motif", motif_id)
        if SLO_COST_PREFIX_SEMANTIC_GROUP == "prefix":
            prefix_hash = str(
                getattr(hint, "prompt_prefix_hash", None)
                or getattr(hint, "prefix_key", None)
                or "unknown_prefix"
            )
            return ("motif", motif_id, stage_id, prefix_hash)
        return ("motif", motif_id, stage_id)

    @staticmethod
    def _semantic_static_prefix_len(req: Req) -> float:
        hint = getattr(req, "structured_hints", None)
        value = getattr(hint, "static_prefix_len", None) if hint is not None else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
        prefix_indices = getattr(req, "prefix_indices", None)
        try:
            return float(len(prefix_indices))
        except (TypeError, ValueError):
            return float(len(getattr(req, "origin_input_ids", None) or []))

    @staticmethod
    def _safe_len(value) -> int:
        if value is None:
            return 0
        numel = getattr(value, "numel", None)
        if callable(numel):
            try:
                return int(numel())
            except (TypeError, ValueError, RuntimeError):
                return 0
        try:
            return len(value)
        except (TypeError, ValueError, RuntimeError):
            return 0

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        if value is None:
            return default
        numel = getattr(value, "numel", None)
        item = getattr(value, "item", None)
        if callable(numel) and callable(item):
            try:
                if int(numel()) == 0:
                    return default
                value = item()
            except (TypeError, ValueError, RuntimeError):
                return default
        try:
            return int(value)
        except (TypeError, ValueError, RuntimeError):
            return default

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if value is None or isinstance(value, bool):
            return default
        numel = getattr(value, "numel", None)
        item = getattr(value, "item", None)
        if callable(numel) and callable(item):
            try:
                if int(numel()) == 0:
                    return default
                value = item()
            except (TypeError, ValueError, RuntimeError):
                return default
        try:
            number = float(value)
        except (TypeError, ValueError, RuntimeError):
            return default
        if math.isnan(number) or math.isinf(number):
            return default
        return number

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_edf_like(waiting_queue: List[Req]) -> None:
        """Sort by remaining SLO slack without using prefix locality.

        This is an intentionally simple deadline baseline for experiments. It
        shares the same React/Motif budget interpretation as slo-prefix-dfs but
        does not look at radix subtrees, so any locality difference comes from
        the policy rather than cache identity changes.
        """
        now = time.perf_counter()
        waiting_queue.sort(
            key=lambda req: (
                SchedulePolicy._slo_prefix_slack_s(req, now),
                getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0),
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")

    @staticmethod
    def _sort_by_structured_hints(
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sort requests by advisory structured hints with balanced group fill.

        The initial structured-hint policy anchored on one running/waiting hint
        and moved all matching requests to the front. That maximizes locality for
        one prefix, but it can also starve other prefixes. Here we group requests
        by their strongest locality hint, emit a capped FIFO run from each group,
        and round-robin across groups.
        """
        if len(waiting_queue) <= 1:
            return

        running_hints = [
            getattr(req, "structured_hints", None)
            for req in (running_batch.reqs if running_batch is not None else [])
            if getattr(req, "structured_hints", None) is not None
        ]
        if running_hints:
            anchors = running_hints
        else:
            first_hint = next(
                (
                    getattr(req, "structured_hints", None)
                    for req in waiting_queue
                    if getattr(req, "structured_hints", None) is not None
                ),
                None,
            )
            anchors = [first_hint] if first_hint is not None else []

        if not anchors and not enable_priority_scheduling:
            return

        indexed_reqs = list(enumerate(waiting_queue))
        has_structured_hints = any(
            getattr(req, "structured_hints", None) is not None
            for _index, req in indexed_reqs
        )

        def hint_value(hint, field_name: str):
            return getattr(hint, field_name, None) if hint is not None else None

        def same_non_empty(candidate, anchor, field_name: str) -> bool:
            candidate_value = hint_value(candidate, field_name)
            return bool(
                candidate_value and candidate_value == hint_value(anchor, field_name)
            )

        def locality_tier_to_anchors(req: Req) -> int:
            hint = getattr(req, "structured_hints", None)
            if hint is None or not anchors:
                return 5
            best = 5
            for anchor in anchors:
                if same_non_empty(hint, anchor, "prefix_key"):
                    best = min(best, 0)
                elif same_non_empty(hint, anchor, "cache_affinity_key"):
                    best = min(best, 1)
                elif same_non_empty(hint, anchor, "decode_class") or same_non_empty(
                    hint, anchor, "grammar_id"
                ):
                    best = min(best, 2)
                elif same_non_empty(hint, anchor, "max_tokens_bucket"):
                    best = min(best, 3)
                elif same_non_empty(hint, anchor, "priority_class"):
                    best = min(best, 4)
            return best

        def group_key(req: Req, original_index: int):
            hint = getattr(req, "structured_hints", None)
            if hint is None:
                # Keep no-hint requests FCFS-compatible. Grouping all no-hint
                # requests together would accidentally create a new policy.
                return ("no_hint", original_index)
            prefix_key = hint_value(hint, "prefix_key")
            if prefix_key:
                return ("prefix", prefix_key)
            cache_affinity_key = hint_value(hint, "cache_affinity_key")
            if cache_affinity_key:
                return ("cache", cache_affinity_key)
            grammar_id = hint_value(hint, "grammar_id")
            decode_class = hint_value(hint, "decode_class")
            if grammar_id or decode_class:
                return ("decode", grammar_id or "", decode_class or "")
            max_tokens_bucket = hint_value(hint, "max_tokens_bucket")
            if max_tokens_bucket:
                return ("max_tokens", max_tokens_bucket)
            priority_class = hint_value(hint, "priority_class")
            if priority_class:
                return ("priority", priority_class)
            return ("hint", original_index)

        def arrival_time(req: Req):
            return getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0)

        def deadline(req: Req):
            hint = getattr(req, "structured_hints", None)
            value = getattr(hint, "deadline_ms", None) if hint is not None else None
            return value if value is not None else float("inf")

        def priority(req: Req):
            if not enable_priority_scheduling:
                return 0
            return (getattr(req, "priority", 0) or 0) * priority_sign

        if not has_structured_hints:
            indexed_reqs.sort(
                key=lambda item: (
                    priority(item[1]),
                    deadline(item[1]),
                    arrival_time(item[1]),
                    item[0],
                )
            )
            waiting_queue[:] = [req for _index, req in indexed_reqs]
            return

        grouped_reqs = defaultdict(list)
        group_meta = {}
        for original_index, req in indexed_reqs:
            key = group_key(req, original_index)
            grouped_reqs[key].append((original_index, req))
            meta = group_meta.setdefault(
                key,
                {
                    "best_tier": locality_tier_to_anchors(req),
                    "first_arrival": arrival_time(req),
                    "first_index": original_index,
                },
            )
            meta["best_tier"] = min(meta["best_tier"], locality_tier_to_anchors(req))
            meta["first_arrival"] = min(meta["first_arrival"], arrival_time(req))
            meta["first_index"] = min(meta["first_index"], original_index)

        for items in grouped_reqs.values():
            items.sort(
                key=lambda item: (
                    priority(item[1]),
                    deadline(item[1]),
                    arrival_time(item[1]),
                    item[0],
                )
            )

        group_order = sorted(
            grouped_reqs,
            key=lambda key: (
                group_meta[key]["best_tier"],
                group_meta[key]["first_arrival"],
                group_meta[key]["first_index"],
            ),
        )

        cap = max(1, STRUCTURED_HINT_GROUP_CAP)
        emitted: List[Req] = []
        while group_order:
            next_group_order = []
            for key in group_order:
                items = grouped_reqs[key]
                for _ in range(min(cap, len(items))):
                    emitted.append(items.pop(0)[1])
                if items:
                    next_group_order.append(key)
            group_order = next_group_order

        if STRUCTURED_HINT_POLICY_DEBUG_LOG:
            prefix_counts = Counter()
            for req in emitted:
                hint = getattr(req, "structured_hints", None)
                prefix_counts[hint_value(hint, "prefix_key") or "<none>"] += 1
            logger.info(
                "structured_hint_balanced_order cap=%s prefix_counts=%s order=%s",
                cap,
                dict(prefix_counts),
                [req.rid for req in emitted],
            )

        waiting_queue[:] = emitted

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _calc_slo_boosted_score(
        cur_node: TreeNode,
        depth: int,
        node_to_weight: Dict[TreeNode, int],
        node_to_react_count: Dict[TreeNode, int],
        node_to_motif_count: Dict[TreeNode, int],
        node_to_score: Dict[TreeNode, float],
        react_risk: float,
        motif_lag: float,
    ) -> None:
        for child in cur_node.children.values():
            child_depth = depth + len(child.key or [])
            SchedulePolicy._calc_slo_boosted_score(
                child,
                child_depth,
                node_to_weight,
                node_to_react_count,
                node_to_motif_count,
                node_to_score,
                react_risk,
                motif_lag,
            )
            node_to_weight[cur_node] += node_to_weight[child]
            node_to_react_count[cur_node] += node_to_react_count[child]
            node_to_motif_count[cur_node] += node_to_motif_count[child]

        weight = node_to_weight[cur_node]
        if weight <= 0:
            node_to_score[cur_node] = 0.0
            return

        react_fraction = node_to_react_count[cur_node] / weight
        motif_fraction = node_to_motif_count[cur_node] / weight
        boost = (
            1.0
            + max(0.0, SLO_BOOST_REACT_ALPHA) * react_risk * react_fraction
            + max(0.0, SLO_BOOST_MOTIF_BETA) * motif_lag * motif_fraction
        )
        boost = min(max(1.0, boost), max(1.0, SLO_BOOST_MAX_MULTIPLIER))
        node_to_score[cur_node] = max(0, depth) * weight * boost

    @staticmethod
    def _calc_slo_prefix_score(
        cur_node: TreeNode,
        depth: int,
        node_to_weight: Dict[TreeNode, int],
        node_to_slack_sum: Dict[TreeNode, float],
        node_to_score: Dict[TreeNode, float],
    ) -> None:
        for child in cur_node.children.values():
            child_depth = depth + len(child.key or [])
            SchedulePolicy._calc_slo_prefix_score(
                child,
                child_depth,
                node_to_weight,
                node_to_slack_sum,
                node_to_score,
            )
            node_to_weight[cur_node] += node_to_weight[child]
            node_to_slack_sum[cur_node] += node_to_slack_sum[child]

        weight = node_to_weight[cur_node]
        if weight <= 0:
            node_to_score[cur_node] = 0.0
            return
        slack_sum = max(SLO_PREFIX_EPS_MS / 1000.0, node_to_slack_sum[cur_node])
        slack_gamma = max(0.0, SLO_PREFIX_SLACK_GAMMA)
        node_to_score[cur_node] = (max(0, depth) * weight) / (
            slack_sum**slack_gamma
        )

    @staticmethod
    def _sort_by_slo_marginal_prefix_dfs(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sort by SLO-prefix pressure with a weak marginal prefill cost penalty.

        This policy keeps the proven slo-prefix-dfs shape and only adds a weak
        denominator based on uncached prefill tokens:

            score(u) =
                prefix_len(u) * waiting_count(u)
                / sum(slack_i)^slack_gamma
                / avg_marginal_prefill_cost(u)^cost_gamma

        With cost_gamma=0, it degenerates to slo-prefix-dfs. The default
        cost_gamma is intentionally small because marginal cost is a correction,
        not the primary objective; radix locality should remain the main signal.
        """

        now = time.perf_counter()
        last_node_to_reqs = defaultdict(list)
        node_to_slack_sum = defaultdict(float)
        node_to_marginal_cost_sum = defaultdict(float)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)
            node_to_slack_sum[req.last_node] += SchedulePolicy._slo_prefix_slack_s(
                req, now
            )
            node_to_marginal_cost_sum[
                req.last_node
            ] += SchedulePolicy._slo_prefix_marginal_cost(req)

        node_to_weight = defaultdict(int)
        node_to_score = defaultdict(float)
        for node, reqs in last_node_to_reqs.items():
            node_to_weight[node] = len(reqs)

        SchedulePolicy._calc_slo_marginal_prefix_score(
            tree_cache.root_node,
            depth=0,
            node_to_weight=node_to_weight,
            node_to_slack_sum=node_to_slack_sum,
            node_to_marginal_cost_sum=node_to_marginal_cost_sum,
            node_to_score=node_to_score,
        )

        waiting_queue.clear()
        SchedulePolicy._get_slo_prefix_dfs_priority(
            tree_cache.root_node,
            node_to_score,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
            now,
        )

    @staticmethod
    def _calc_slo_marginal_prefix_score(
        cur_node: TreeNode,
        depth: int,
        node_to_weight: Dict[TreeNode, int],
        node_to_slack_sum: Dict[TreeNode, float],
        node_to_marginal_cost_sum: Dict[TreeNode, float],
        node_to_score: Dict[TreeNode, float],
    ) -> None:
        for child in cur_node.children.values():
            child_depth = depth + len(child.key or [])
            SchedulePolicy._calc_slo_marginal_prefix_score(
                child,
                child_depth,
                node_to_weight,
                node_to_slack_sum,
                node_to_marginal_cost_sum,
                node_to_score,
            )
            node_to_weight[cur_node] += node_to_weight[child]
            node_to_slack_sum[cur_node] += node_to_slack_sum[child]
            node_to_marginal_cost_sum[cur_node] += node_to_marginal_cost_sum[child]

        weight = node_to_weight[cur_node]
        if weight <= 0:
            node_to_score[cur_node] = 0.0
            return

        slack_sum = max(SLO_PREFIX_EPS_MS / 1000.0, node_to_slack_sum[cur_node])
        slack_gamma = max(0.0, SLO_PREFIX_SLACK_GAMMA)
        avg_marginal_cost = max(
            1.0, node_to_marginal_cost_sum[cur_node] / max(1, weight)
        )
        cost_gamma = max(0.0, SLO_PREFIX_MARGINAL_COST_GAMMA)
        node_to_score[cur_node] = (max(0, depth) * weight) / (
            (slack_sum**slack_gamma) * (avg_marginal_cost**cost_gamma)
        )

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        children = [child for child in cur_node.children.values()]
        children.sort(key=lambda x: -node_to_priority[x])
        for child in children:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])

    @staticmethod
    def _get_slo_prefix_dfs_priority(
        cur_node: TreeNode,
        node_to_score: Dict[TreeNode, float],
        node_to_weight: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
        now: float,
    ) -> None:
        children = [child for child in cur_node.children.values()]
        children.sort(
            key=lambda x: (
                -node_to_score[x],
                -node_to_weight[x],
            )
        )
        for child in children:
            SchedulePolicy._get_slo_prefix_dfs_priority(
                child,
                node_to_score,
                node_to_weight,
                last_node_to_reqs,
                q,
                now,
            )
        reqs = last_node_to_reqs[cur_node]
        reqs.sort(key=lambda req: SchedulePolicy._slo_prefix_slack_s(req, now))
        q.extend(reqs)

    @staticmethod
    def _sort_by_slo_cost_prefix_dfs(
        waiting_queue: List[Req],
        tree_cache: BasePrefixCache,
        *,
        semantic_overlay: bool,
    ) -> None:
        """Sort by reuse value, SLO urgency, and estimated prompt cost.

        Pure mode scores each real radix subtree:

            score(u) = max(eps, prefix_len(u) * (waiting_count(u) - 1))
                     * sum(exp(-slack_i / tau_i))
                     / sum(prompt_cost_i)

        Semantic mode keeps the same radix cache identity, but adds a virtual
        motif/stage pressure term to every radix subtree containing requests
        from that semantic group. This is the semantic-fragmentation hook: a
        Motif stage split across several small radix leaves can still build
        group-level urgency, without pretending that different token prefixes
        are the same cache key.
        """
        now = time.perf_counter()
        last_node_to_reqs = defaultdict(list)
        node_to_weight = defaultdict(int)
        node_to_urgency = defaultdict(float)
        node_to_cost = defaultdict(float)
        node_to_score = defaultdict(float)
        node_to_semantic_keys = defaultdict(set)
        semantic_stats = {}

        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)
            urgency = SchedulePolicy._slo_cost_urgency(req, now)
            cost = SchedulePolicy._slo_cost_service_cost(req)
            node_to_weight[req.last_node] += 1
            node_to_urgency[req.last_node] += urgency
            node_to_cost[req.last_node] += cost
            if semantic_overlay:
                semantic_key = SchedulePolicy._semantic_fragment_key(req)
                if semantic_key is not None:
                    node_to_semantic_keys[req.last_node].add(semantic_key)
                    stats = semantic_stats.setdefault(
                        semantic_key,
                        {
                            "count": 0,
                            "urgency": 0.0,
                            "cost": 0.0,
                            "prefix_len": float("inf"),
                        },
                    )
                    stats["count"] += 1
                    stats["urgency"] += urgency
                    stats["cost"] += cost
                    stats["prefix_len"] = min(
                        stats["prefix_len"],
                        SchedulePolicy._semantic_static_prefix_len(req),
                    )

        semantic_group_score = {}
        if semantic_overlay:
            for key, stats in semantic_stats.items():
                count = int(stats["count"])
                prefix_len = stats["prefix_len"]
                if prefix_len == float("inf"):
                    prefix_len = 0.0
                reuse_value = max(0.0, float(prefix_len)) * max(0, count - 1)
                semantic_group_score[key] = (
                    reuse_value
                    * float(stats["urgency"])
                    / max(1.0, float(stats["cost"]))
                )

        SchedulePolicy._calc_slo_cost_prefix_score(
            tree_cache.root_node,
            depth=0,
            node_to_weight=node_to_weight,
            node_to_urgency=node_to_urgency,
            node_to_cost=node_to_cost,
            node_to_score=node_to_score,
            node_to_semantic_keys=node_to_semantic_keys,
            semantic_group_score=semantic_group_score,
            semantic_overlay=semantic_overlay,
        )

        waiting_queue.clear()
        SchedulePolicy._get_slo_cost_prefix_dfs_priority(
            tree_cache.root_node,
            node_to_score,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
            now,
        )

    @staticmethod
    def _calc_slo_cost_prefix_score(
        cur_node: TreeNode,
        depth: int,
        node_to_weight: Dict[TreeNode, int],
        node_to_urgency: Dict[TreeNode, float],
        node_to_cost: Dict[TreeNode, float],
        node_to_score: Dict[TreeNode, float],
        node_to_semantic_keys: Dict[TreeNode, Set[tuple]],
        semantic_group_score: Dict[tuple, float],
        semantic_overlay: bool,
    ) -> None:
        for child in cur_node.children.values():
            child_depth = depth + len(child.key or [])
            SchedulePolicy._calc_slo_cost_prefix_score(
                child,
                child_depth,
                node_to_weight,
                node_to_urgency,
                node_to_cost,
                node_to_score,
                node_to_semantic_keys,
                semantic_group_score,
                semantic_overlay,
            )
            node_to_weight[cur_node] += node_to_weight[child]
            node_to_urgency[cur_node] += node_to_urgency[child]
            node_to_cost[cur_node] += node_to_cost[child]
            if semantic_overlay and node_to_semantic_keys[child]:
                node_to_semantic_keys[cur_node].update(node_to_semantic_keys[child])

        weight = node_to_weight[cur_node]
        if weight <= 0:
            node_to_score[cur_node] = 0.0
            return

        reuse_value = max(
            SLO_COST_PREFIX_MIN_REUSE_VALUE,
            max(0, depth) * max(0, weight - 1),
        )
        prefix_score = (
            float(reuse_value)
            * float(node_to_urgency[cur_node])
            / max(1.0, float(node_to_cost[cur_node]))
        )
        semantic_score = 0.0
        if semantic_overlay:
            semantic_score = SLO_COST_PREFIX_SEMANTIC_WEIGHT * sum(
                semantic_group_score.get(key, 0.0)
                for key in node_to_semantic_keys[cur_node]
            )
        node_to_score[cur_node] = prefix_score + semantic_score

    @staticmethod
    def _get_slo_cost_prefix_dfs_priority(
        cur_node: TreeNode,
        node_to_score: Dict[TreeNode, float],
        node_to_weight: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
        now: float,
    ) -> None:
        children = [child for child in cur_node.children.values()]
        children.sort(
            key=lambda x: (
                -node_to_score[x],
                -node_to_weight[x],
            )
        )
        for child in children:
            SchedulePolicy._get_slo_cost_prefix_dfs_priority(
                child,
                node_to_score,
                node_to_weight,
                last_node_to_reqs,
                q,
                now,
            )
        reqs = last_node_to_reqs[cur_node]
        reqs.sort(
            key=lambda req: (
                SchedulePolicy._slo_prefix_raw_slack_s(req, now),
                SchedulePolicy._slo_cost_service_cost(req),
                getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0),
            )
        )
        q.extend(reqs)

    def _sort_by_slo_prefix_mixed_dfs(
        self, waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Keep DFS as the main lane and cache a bounded SLO-prefix slice order.

        The scheduler uses the cached SLO-prefix order at prefill batch
        construction time, so the split applies to the current serving batch
        instead of globally interleaving the entire waiting queue.
        """
        self.slo_prefix_mixed_slo_order_rids = []
        if len(waiting_queue) <= 1:
            return

        dfs_order = list(waiting_queue)
        slo_order = list(waiting_queue)
        SchedulePolicy._sort_by_dfs_weight(dfs_order, tree_cache)
        SchedulePolicy._sort_by_slo_prefix_dfs(slo_order, tree_cache)
        self.slo_prefix_mixed_slo_order_rids = [req.rid for req in slo_order]
        waiting_queue[:] = dfs_order

    def build_slo_prefix_mixed_batch_order(
        self, waiting_queue: List[Req]
    ) -> Optional[List[Req]]:
        """Return a per-batch DFS/SLO candidate order for mixed scheduling."""

        policy_value = getattr(getattr(self, "policy", None), "value", None)
        if policy_value != "slo-prefix-mixed-dfs" or len(waiting_queue) <= 1:
            return None

        dfs_slots = max(1, SLO_PREFIX_MIX_DFS_SLOTS)
        slo_slots = max(0, SLO_PREFIX_MIX_SLO_SLOTS)
        if slo_slots <= 0 or not self.slo_prefix_mixed_slo_order_rids:
            return None

        rid_to_req = {req.rid: req for req in waiting_queue}
        slo_order = [
            rid_to_req[rid]
            for rid in self.slo_prefix_mixed_slo_order_rids
            if rid in rid_to_req
        ]
        pattern = ["dfs"] * dfs_slots + ["slo"] * slo_slots
        selected = set()
        mixed_order: List[Req] = []
        dfs_idx = 0
        slo_idx = 0

        def take_from(order: List[Req], start: int) -> tuple[Optional[Req], int]:
            idx = start
            while idx < len(order):
                req = order[idx]
                idx += 1
                if req.rid not in selected:
                    return req, idx
            return None, idx

        while len(mixed_order) < len(waiting_queue):
            for lane in pattern:
                if len(mixed_order) >= len(waiting_queue):
                    break
                if lane == "slo":
                    req, slo_idx = take_from(slo_order, slo_idx)
                    if req is None:
                        req, dfs_idx = take_from(waiting_queue, dfs_idx)
                else:
                    req, dfs_idx = take_from(waiting_queue, dfs_idx)
                    if req is None:
                        req, slo_idx = take_from(slo_order, slo_idx)
                if req is None:
                    continue
                selected.add(req.rid)
                mixed_order.append(req)

        return mixed_order


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        mixed_with_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens
        self.rem_total_token_offset = mixed_with_decode_tokens
        self.cur_rem_token_offset = mixed_with_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0

        if running_batch is not None:
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.nsa_prefill_cp_in_seq_split = is_nsa_prefill_cp_in_seq_split()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        if self.is_hybrid_swa:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size(),
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self):
        if self.is_hybrid_swa:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size(),
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        self.rem_total_token_offset += extend_input_len + max_new_tokens
        self.cur_rem_token_offset += extend_input_len
        self.rem_input_tokens -= extend_input_len

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def _get_dllm_remain_tokens(self) -> int:
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            int(self.rem_total_tokens),
        )
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # Make sure at least one page is available
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        req.extend_input_len = trunc_len
        req.fill_ids = req.fill_ids[: prefix_len + trunc_len]

        self.can_run_list.append(req)

        self._update_prefill_budget(prefix_len, trunc_len, 0)

    def _req_inc_lock_ref(self, req: Req):
        if self.is_hybrid_swa:
            swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
            req.swa_uuid_for_lock = swa_uuid_for_lock
        else:
            self.tree_cache.inc_lock_ref(req.last_node)

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                _rem_tokens = self.rem_chunk_tokens

        truncated = req.extend_input_len > _rem_tokens
        req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        try:
            if self.tree_cache.supports_swa() and self.tree_cache.is_tree_cache():
                swa_uuid_for_lock = self.tree_cache.inc_lock_ref(last_node)
            else:
                self.tree_cache.inc_lock_ref(last_node)
            yield None
        finally:
            if self.tree_cache.supports_swa() and self.tree_cache.is_tree_cache():
                self.tree_cache.dec_lock_ref(last_node, swa_uuid_for_lock)
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        # Early exit if no enough tokens for the input tokens
        if self.ceil_paged_tokens(req.extend_input_len) > min(
            self.cur_rem_tokens, self.rem_total_tokens
        ):
            return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(
                req.extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            # Non-chunked prefill
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            req.set_extend_input_len(trunc_len)
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(0, trunc_len, 0)

        return self.budget_state()

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        if self.nsa_prefill_cp_in_seq_split and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        total_tokens = req.extend_input_len + min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = req.extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if req.host_hit_length > 0:
                load_back_start = time.perf_counter()
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    req.last_host_node, req.host_hit_length
                )
                req.load_back_submit_wall_ms = (
                    time.perf_counter() - load_back_start
                ) * 1000.0
                req.load_back_tokens = int(new_indices.numel())
                req.load_back_submitted = req.load_back_tokens > 0
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                return AddReqResult.OTHER

            if (self.prefill_delayer_single_pass is not None) and (
                not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                    local_prefillable=True
                )
            ):
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                )
            else:
                # Make sure at least one page is available
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                # Chunked prefill
                req.set_extend_input_len(trunc_len)
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]

                self.can_run_list.append(req)
                self.new_chunked_req = req

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(prefix_len, trunc_len, 0)

        return self.budget_state()

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) check_finished + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        min_tokens_to_remove = (
            req.extend_input_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    break
            else:
                break

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                self.rem_total_token_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
