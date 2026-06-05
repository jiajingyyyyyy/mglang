from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclasses.dataclass
class PLASProgramState:
    program_id: str
    completed_service_s: float = 0.0
    completed_wait_s: float = 0.0
    num_completed_reqs: int = 0
    num_active_reqs: int = 0
    first_seen_time: float = 0.0
    last_arrival_time: float = 0.0
    last_finish_time: float = 0.0


@dataclasses.dataclass(frozen=True)
class PLASQueueConfig:
    service_thresholds_s: Tuple[float, ...]
    quanta_s: Tuple[float, ...]
    starvation_beta: float


class PLASProcessTable:
    """Process table for Program-Level Attained Service scheduling.

    PLAS gives higher admission priority to requests whose program has received
    less completed model service so far. The table is intentionally small and
    scheduler-local; multi-engine routing and ATLAS are outside this layer.
    """

    def __init__(
        self,
        missing_program_id_policy: str = "rid",
        queue_config: Optional[PLASQueueConfig] = None,
    ):
        if missing_program_id_policy not in {"rid", "session", "error"}:
            raise ValueError(
                "missing_program_id_policy must be one of: rid, session, error"
            )
        self.missing_program_id_policy = missing_program_id_policy
        self.queue_config = queue_config
        self.programs: Dict[str, PLASProgramState] = {}
        self.missing_program_id_count = 0
        self.demoted_count = 0
        self.promoted_count = 0
        self.preempted_count = 0

    def get_program_id(self, req: "Req") -> str:
        hint = getattr(req, "structured_hints", None)
        for field_name in ("program_id", "task_id"):
            value = getattr(hint, field_name, None) if hint is not None else None
            if isinstance(value, str) and value:
                return value

        self.missing_program_id_count += 1
        if self.missing_program_id_policy == "error":
            raise ValueError(
                f"PLAS request {getattr(req, 'rid', None)} is missing program_id/task_id"
            )

        session_id = getattr(req, "session_id", None)
        if self.missing_program_id_policy == "session":
            if isinstance(session_id, str) and session_id:
                return session_id
            raise ValueError(
                f"PLAS request {getattr(req, 'rid', None)} is missing session_id"
            )

        if isinstance(session_id, str) and session_id:
            return session_id
        return str(getattr(req, "rid", "unknown"))

    def _state_for(
        self, program_id: str, now: Optional[float] = None
    ) -> PLASProgramState:
        state = self.programs.get(program_id)
        if state is None:
            seen_time = float(now or 0.0)
            state = PLASProgramState(
                program_id=program_id,
                first_seen_time=seen_time,
                last_arrival_time=seen_time,
            )
            self.programs[program_id] = state
        return state

    def on_enqueue(
        self, req: "Req", now: float, *, is_retracted: bool = False
    ) -> PLASProgramState:
        program_id = self.get_program_id(req)
        req.plas_program_id = program_id
        req.plas_enqueue_time = now
        req.plas_last_queue_enter_time = now

        state = self._state_for(program_id, now)
        if state.first_seen_time <= 0.0:
            state.first_seen_time = now
        state.last_arrival_time = now
        if self.queue_config is not None and not is_retracted:
            self.assign_initial_queue(req, state.completed_service_s)
        if not getattr(req, "plas_active_counted", False):
            state.num_active_reqs += 1
            req.plas_active_counted = True
        return state

    def priority(self, req: "Req") -> Tuple[float, float, str]:
        program_id = getattr(req, "plas_program_id", None)
        if not program_id:
            program_id = self.get_program_id(req)
            req.plas_program_id = program_id
        state = self._state_for(program_id)
        wait_entry = getattr(
            getattr(req, "time_stats", None), "wait_queue_entry_time", 0
        )
        if not wait_entry:
            wait_entry = getattr(req, "plas_enqueue_time", 0.0) or 0.0
        return (state.completed_service_s, float(wait_entry), str(req.rid))

    def mlfq_priority(self, req: "Req") -> Tuple[int, float, str]:
        queue_idx = getattr(req, "plas_queue_idx", None)
        if queue_idx is None:
            program_id = getattr(req, "plas_program_id", None) or self.get_program_id(
                req
            )
            req.plas_program_id = program_id
            state = self._state_for(program_id)
            self.assign_initial_queue(req, state.completed_service_s)
            queue_idx = getattr(req, "plas_queue_idx", 0)
        queue_enter = getattr(req, "plas_last_queue_enter_time", 0.0) or getattr(
            req, "plas_enqueue_time", 0.0
        )
        return (int(queue_idx), float(queue_enter or 0.0), str(req.rid))

    def on_request_service(
        self, req: "Req", service_s: float, now: float, wait_s: float = 0.0
    ) -> PLASProgramState:
        program_id = getattr(req, "plas_program_id", None) or self.get_program_id(req)
        req.plas_program_id = program_id
        state = self._state_for(program_id, now)
        service_s = max(0.0, float(service_s or 0.0))
        wait_s = max(0.0, float(wait_s or 0.0))
        state.completed_service_s += service_s
        state.completed_wait_s += wait_s
        state.num_completed_reqs += 1
        if getattr(req, "plas_active_counted", False):
            state.num_active_reqs = max(0, state.num_active_reqs - 1)
            req.plas_active_counted = False
        state.last_finish_time = now
        return state

    def assign_initial_queue(self, req: "Req", service_s: float) -> int:
        queue_idx = self.queue_index_for_service(service_s)
        req.plas_queue_idx = queue_idx
        req.plas_priority_at_admit = float(service_s)
        req.plas_quanta_s = self.quantum_for_queue(queue_idx)
        return queue_idx

    def queue_index_for_service(self, service_s: float) -> int:
        if self.queue_config is None:
            return 0
        service_s = max(0.0, float(service_s or 0.0))
        thresholds = self.queue_config.service_thresholds_s
        queue_idx = 0
        for idx, lower_bound in enumerate(thresholds):
            if service_s >= lower_bound:
                queue_idx = idx
            else:
                break
        return min(queue_idx, len(thresholds) - 1)

    def quantum_for_queue(self, queue_idx: int) -> float:
        if self.queue_config is None:
            return 0.0
        queue_idx = max(0, min(int(queue_idx), len(self.queue_config.quanta_s) - 1))
        return max(0.0, float(self.queue_config.quanta_s[queue_idx]))

    def on_admit(self, req: "Req", now: float) -> None:
        self.add_wait_time_since_queue_enter(req, now)
        if getattr(req, "plas_quanta_s", 0.0) <= 0.0:
            req.plas_quanta_s = self.quantum_for_queue(
                getattr(req, "plas_queue_idx", 0)
            )

    def add_wait_time_since_queue_enter(self, req: "Req", now: float) -> float:
        queue_enter = getattr(req, "plas_last_queue_enter_time", 0.0) or 0.0
        if queue_enter <= 0.0:
            return 0.0
        delta = max(0.0, now - queue_enter)
        req.plas_call_wait_s += delta
        req.plas_last_queue_enter_time = 0.0
        return delta

    def on_forward_step(self, req: "Req", elapsed_s: float) -> None:
        elapsed_s = max(0.0, float(elapsed_s or 0.0))
        if elapsed_s <= 0.0:
            return
        req.plas_call_model_time_s += elapsed_s
        req.plas_quanta_s = max(0.0, float(req.plas_quanta_s or 0.0) - elapsed_s)

    def maybe_demote(self, req: "Req") -> bool:
        if self.queue_config is None or getattr(req, "plas_quanta_s", 0.0) > 0.0:
            return False
        old_idx = int(getattr(req, "plas_queue_idx", 0) or 0)
        new_idx = min(old_idx + 1, len(self.queue_config.service_thresholds_s) - 1)
        req.plas_queue_idx = new_idx
        req.plas_quanta_s = self.quantum_for_queue(new_idx)
        if new_idx != old_idx:
            self.demoted_count += 1
            req.plas_demoted_count += 1
        return True

    def maybe_promote_for_starvation(self, req: "Req", now: float) -> bool:
        if self.queue_config is None:
            return False
        program_id = getattr(req, "plas_program_id", None) or self.get_program_id(req)
        state = self._state_for(program_id, now)
        queue_wait = 0.0
        if getattr(req, "plas_last_queue_enter_time", 0.0):
            queue_wait = max(0.0, now - req.plas_last_queue_enter_time)
        total_wait = (
            state.completed_wait_s
            + float(getattr(req, "plas_call_wait_s", 0.0) or 0.0)
            + queue_wait
        )
        total_service = (
            state.completed_service_s
            + float(getattr(req, "plas_call_model_time_s", 0.0) or 0.0)
        )
        if total_service <= 0.0:
            return False
        req.plas_starvation_ratio = total_wait / total_service
        if req.plas_starvation_ratio < self.queue_config.starvation_beta:
            return False

        was_lower_queue = int(getattr(req, "plas_queue_idx", 0) or 0) > 0
        req.plas_queue_idx = 0
        req.plas_quanta_s = self.quantum_for_queue(0)
        # Algorithm 1 resets only the call-local wait/model counters.
        req.plas_call_wait_s = 0.0
        req.plas_call_model_time_s = 0.0
        if getattr(req, "plas_last_queue_enter_time", 0.0):
            req.plas_last_queue_enter_time = now
        if was_lower_queue:
            self.promoted_count += 1
            req.plas_boosted_count += 1
        return was_lower_queue

    def on_request_abort(self, req: "Req", now: float) -> None:
        program_id = getattr(req, "plas_program_id", None)
        if not program_id:
            return
        state = self._state_for(program_id, now)
        if getattr(req, "plas_active_counted", False):
            state.num_active_reqs = max(0, state.num_active_reqs - 1)
            req.plas_active_counted = False
        state.last_finish_time = now

    def program_service(self, program_id: Optional[str]) -> float:
        if not program_id:
            return 0.0
        state = self.programs.get(program_id)
        return float(state.completed_service_s) if state is not None else 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "missing_program_id_count": self.missing_program_id_count,
            "demoted_count": self.demoted_count,
            "promoted_count": self.promoted_count,
            "preempted_count": self.preempted_count,
            "programs": {
                program_id: dataclasses.asdict(state)
                for program_id, state in self.programs.items()
            },
        }
