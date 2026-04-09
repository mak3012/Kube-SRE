from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from pydantic import BaseModel

from .models import ClusterTelemetry, EventSnippet, KubeSREObservation, KubeToolAction, PodStatus, ToolName

def _now() -> float:
    return time.time()

@dataclass
class _Pod:
    name: str
    namespace: str
    image: str
    phase: str = "Running"
    ready: bool = True
    restarts: int = 0
    reason: Optional[str] = None
    message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    mem_limit_mib: int = 256
    mem_request_mib: int = 128
    mem_usage_mib: int = 64

@dataclass
class _Deployment:
    name: str
    namespace: str
    image: str = "demo/web:v1"
    mem_limit_mib: int = 256
    mem_request_mib: int = 128

@dataclass
class _VirtualService:
    name: str
    namespace: str
    host: str
    route_ok: bool = True

class ScenarioSpec(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str

SCENARIOS: Dict[str, ScenarioSpec] = {
    "ghost_image": ScenarioSpec(
        id="ghost_image",
        title="The Ghost Image",
        difficulty="easy",
        description="Fix ImagePullBackOff caused by a typo in the image tag.",
    ),
    "memory_leak": ScenarioSpec(
        id="memory_leak",
        title="The Memory Leak",
        difficulty="medium",
        description="Resolve OOMKilled by adjusting memory requests/limits.",
    ),
    "cascading_mesh_failure": ScenarioSpec(
        id="cascading_mesh_failure",
        title="Cascading Mesh Failure",
        difficulty="hard",
        description="Mitigate service mesh connectivity via virtual service patching.",
    ),
}

class KubeSREGymEnv(Environment[KubeToolAction, KubeSREObservation, Dict[str, Any]]):
    """
    openenv-core Environment that simulates Kubernetes incidents.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Score bounds — strictly within (0, 1), never equal to 0.0 or 1.0
    _SCORE_MIN: float = 0.01
    _SCORE_MAX: float = 0.99

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = random.Random(seed)
        self.task_id: str = "ghost_image"
        self.episode_id: Optional[str] = None
        self.step_n: int = 0
        self.reward: float = 0.0
        self.done: bool = False
        self._pods: Dict[Tuple[str, str], _Pod] = {}
        self._deployments: Dict[Tuple[str, str], _Deployment] = {}
        self._virtualservices: Dict[Tuple[str, str], _VirtualService] = {}
        self._events: List[EventSnippet] = []
        self._seen_log_success: bool = False
        self._last_error_distance: float = 1.0
        self._identified_error: bool = False
        self._attempted_fix: bool = False
        self._max_steps: int = 40
        self._episode_start: float = _now()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> KubeSREObservation:
        if seed is not None:
            self._rng.seed(seed)
        self.episode_id = episode_id
        self.task_id = str(kwargs.get("task_id") or kwargs.get("scenario_id") or self.task_id)
        if self.task_id not in SCENARIOS:
            self.task_id = "ghost_image"
        self.step_n = 0
        self.reward = 0.0
        self.done = False
        self._episode_start = _now()
        self._seen_log_success = False
        self._last_error_distance = 1.0
        self._identified_error = False
        self._attempted_fix = False
        self._pods.clear()
        self._deployments.clear()
        self._virtualservices.clear()
        self._events.clear()
        self._bootstrap()
        self._inject_fault()
        return self._obs(text=self._render())

    def step(self, action: KubeToolAction, timeout_s: Optional[float] = None, **kwargs: Any) -> KubeSREObservation:
        if self.done:
            return self._obs(text=self._render(extra="Episode already done; call reset()."))
        self.step_n += 1
        self.reward = -0.05
        ok = True
        err: Optional[str] = None
        out: Any = None
        try:
            a = action.args
            if action.tool == ToolName.get_pods:
                out = self.get_pods(namespace=getattr(a, "namespace", "prod"))
            elif action.tool == ToolName.describe_pod:
                out = self.describe_pod(name=a.name, namespace=getattr(a, "namespace", "prod"))
            elif action.tool == ToolName.logs:
                out = self.logs(pod=a.pod, namespace=getattr(a, "namespace", "prod"), tail_lines=a.tail_lines)
            elif action.tool == ToolName.patch_deployment:
                out = self.patch_deployment(name=a.name, patch=a.patch, namespace=getattr(a, "namespace", "prod"))
            elif action.tool == ToolName.patch_virtualservice:
                out = self.patch_virtualservice(name=a.name, patch=a.patch, namespace=getattr(a, "namespace", "prod"))
            else:
                ok = False
                err = "unknown_tool"
                out = {"error": "unknown_tool"}
        except Exception as e:
            ok = False
            err = e.__class__.__name__
            out = {"error": str(e)}

        if not ok:
            self.reward += -0.2
        self._update_grader_signals(tool=action.tool.value, out=out)
        gamma = 0.99
        phi_prev = self._potential()
        self._tick()
        phi_next = self._potential()
        self.reward += gamma * phi_next - phi_prev

        if action.tool == ToolName.logs and ok and not self._seen_log_success and isinstance(out, dict) and "lines" in out:
            self.reward += 0.1
            self._seen_log_success = True

        dist = self._error_distance()
        if dist < self._last_error_distance - 1e-9:
            self.reward += 0.1
        self._last_error_distance = dist

        if self._is_success():
            self.reward += 1.0
            self.done = True
        elif self._is_failure() or self.step_n >= self._max_steps:
            self.reward += -1.0
            self.done = True

        return self._obs(text=self._render(tool_output=out, error=err))

    # ---- THE PERMANENT FOOLPROOF CLAMP ----
    def _get_safe_score(self) -> float:
        """
        Mathematically enforces the strictly (0, 1) rule under all conditions.
        No raw score value anywhere in this class should ever bypass this method.
        """
        raw_score = self._score()
        return max(self._SCORE_MIN, min(self._SCORE_MAX, float(raw_score)))

    @property
    def state(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "step": self.step_n, "reward": self.reward, "done": self.done, "score": self._get_safe_score()}

    def get_pods(self, namespace: str = "prod") -> Dict[str, Any]:
        pods = [p for (ns, _), p in self._pods.items() if ns == namespace]
        return {"items": [self._pod_to_dict(p) for p in pods]}

    def describe_pod(self, name: str, namespace: str = "prod") -> Dict[str, Any]:
        p = self._pods.get((namespace, name))
        if not p:
            return {"error": "NotFound"}
        return {"pod": self._pod_to_dict(p), "events": [e.model_dump() for e in self._events[-10:]]}

    def logs(self, pod: str, namespace: str = "prod", tail_lines: int = 200) -> Dict[str, Any]:
        p = self._pods.get((namespace, pod))
        if not p:
            return {"error": "NotFound"}
        if p.reason == "ImagePullBackOff":
            return {"error": "Pod not started — no logs available"}
        return {"pod": pod, "lines": p.logs[-tail_lines:]}

    def patch_deployment(self, name: str, patch: Dict[str, Any], namespace: str = "prod") -> Dict[str, Any]:
        d = self._deployments.get((namespace, name))
        if not d:
            return {"error": "NotFound"}
        spec = patch.get("spec", {})
        c0 = spec.get("template", {}).get("spec", {}).get("containers", [{}])[0]
        if isinstance(c0, dict) and isinstance(c0.get("image"), str):
            d.image = c0["image"]
        resources = c0.get("resources", {}) if isinstance(c0, dict) else {}
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})
        if "memoryMiB" in limits:
            d.mem_limit_mib = int(limits["memoryMiB"])
        if "memoryMiB" in requests:
            d.mem_request_mib = int(requests["memoryMiB"])
        self._sync_pod(namespace, d)
        self._attempted_fix = True
        self._events.append(EventSnippet(ts=_now(), involved_object=f"deployment/{name}", reason="Patched", message="Deployment patched"))
        return {"ok": True}

    def patch_virtualservice(self, name: str, patch: Dict[str, Any], namespace: str = "prod") -> Dict[str, Any]:
        v = self._virtualservices.get((namespace, name))
        if not v:
            return {"error": "NotFound"}
        spec = patch.get("spec", {})
        if "routeOk" in spec:
            v.route_ok = bool(spec["routeOk"])
        self._attempted_fix = True
        self._events.append(EventSnippet(ts=_now(), involved_object=f"virtualservice/{name}", reason="Patched", message="VirtualService patched"))
        return {"ok": True}

    def _bootstrap(self) -> None:
        ns = "prod"
        d = _Deployment(name="web", namespace=ns, image="demo/web:v1", mem_request_mib=128, mem_limit_mib=256)
        self._deployments[(ns, d.name)] = d
        p = _Pod(name="web-0", namespace=ns, image=d.image)
        p.logs.extend(["boot: starting web server", "ready: listening on :8080"])
        self._pods[(ns, p.name)] = p
        vs = _VirtualService(name="web", namespace=ns, host="web.prod.svc.cluster.local", route_ok=True)
        self._virtualservices[(ns, vs.name)] = vs

    def _inject_fault(self) -> None:
        ns = "prod"
        p = self._pods[(ns, "web-0")]
        d = self._deployments[(ns, "web")]
        v = self._virtualservices[(ns, "web")]
        if self.task_id == "ghost_image":
            d.image = "demo/web:v1l"
            self._sync_pod(ns, d)
            p.phase = "Pending"
            p.ready = False
            p.reason = "ImagePullBackOff"
            p.message = "manifest unknown"
            self._events.append(EventSnippet(ts=_now(), involved_object="pod/web-0", reason="Failed", message="ImagePullBackOff: manifest unknown"))
        elif self.task_id == "memory_leak":
            p.mem_usage_mib = 310
            p.mem_limit_mib = 256
            p.ready = False
            p.reason = "OOMKilled"
            p.logs.extend(["warn: memory usage climbing", "error: out of memory"])
            self._events.append(EventSnippet(ts=_now(), involved_object="pod/web-0", reason="Killing", message="OOMKilled"))
        elif self.task_id == "cascading_mesh_failure":
            v.route_ok = False
            p.logs.extend(["mesh: upstream connect error", "mesh: retry budget exceeded"])
            self._events.append(EventSnippet(ts=_now(), involved_object="virtualservice/web", reason="Warning", message="Route misconfiguration detected"))

    def _apply_chaos(self, chaos: Dict[str, Any]) -> None:
        if chaos.get("log_noise"):
            p = self._pods.get(("prod", "web-0"))
            if p:
                p.logs.append("debug: retrying request")

    def _sync_pod(self, namespace: str, d: _Deployment) -> None:
        p = self._pods.get((namespace, "web-0"))
        if p:
            p.image = d.image
            p.mem_limit_mib = d.mem_limit_mib
            p.mem_request_mib = d.mem_request_mib

    def _tick(self) -> None:
        ns = "prod"
        p = self._pods.get((ns, "web-0"))
        d = self._deployments.get((ns, "web"))
        v = self._virtualservices.get((ns, "web"))
        if not p or not d or not v:
            return
        if self.task_id == "ghost_image":
            if d.image == "demo/web:v1":
                p.phase = "Running"
                p.ready = True
                p.reason = None
                p.message = None
            else:
                p.phase = "Pending"
                p.ready = False
                p.reason = "ImagePullBackOff"
        elif self.task_id == "memory_leak":
            p.ready = p.mem_limit_mib >= p.mem_usage_mib
            p.reason = None if p.ready else "OOMKilled"
        elif self.task_id == "cascading_mesh_failure":
            if v.route_ok and "mesh: route restored" not in p.logs:
                p.logs.append("mesh: route restored")

    def _error_distance(self) -> float:
        ns = "prod"
        p = self._pods.get((ns, "web-0"))
        v = self._virtualservices.get((ns, "web"))
        if not p:
            return 1.0
        if self.task_id == "ghost_image":
            return 0.0 if (p.ready and p.phase == "Running") else 1.0
        if self.task_id == "memory_leak":
            return 0.0 if p.mem_limit_mib >= p.mem_usage_mib else min(1.0, (p.mem_usage_mib - p.mem_limit_mib) / 256.0)
        if self.task_id == "cascading_mesh_failure":
            return 0.0 if (v and v.route_ok) else 1.0
        return 1.0

    def _potential(self) -> float:
        # Clamp to strictly (0, 1) so reward shaping never produces
        # a phi value of exactly 0.0 or 1.0 that could leak into score.
        raw = 1.0 - self._error_distance()
        return max(self._SCORE_MIN, min(self._SCORE_MAX, raw))

    def _is_success(self) -> bool:
        return self._error_distance() <= 1e-9

    def _is_failure(self) -> bool:
        return _now() - self._episode_start > 60 * 15

    def _score(self) -> float:
        """
        Returns a raw score. All values are already within [0.01, 0.99].
        _get_safe_score() applies a final clamp before any value leaves this class.
        """
        if self._is_success():
            return 0.99   # strictly < 1.0  ✓
        if self.done and self._is_failure():
            return 0.01   # strictly > 0.0  ✓
        score = 0.01
        if self._identified_error:
            score = max(score, 0.50)
        if self._attempted_fix:
            score = max(score, 0.75)
        return score  # 0.01 | 0.50 | 0.75 — all strictly within (0, 1)  ✓

    def _update_grader_signals(self, tool: str, out: Any) -> None:
        s = ""
        try:
            s = json.dumps(out) if isinstance(out, (dict, list)) else str(out)
        except Exception:
            s = str(out)
        if tool in ("describe_pod", "logs"):
            if self.task_id == "ghost_image" and ("ImagePullBackOff" in s or "manifest unknown" in s):
                self._identified_error = True
            if self.task_id == "memory_leak" and ("OOMKilled" in s or "out of memory" in s):
                self._identified_error = True
            if self.task_id == "cascading_mesh_failure" and ("misconfiguration" in s or "routeOk" in s):
                self._identified_error = True
        if tool in ("patch_deployment", "patch_virtualservice"):
            self._attempted_fix = True

    def _pod_to_dict(self, p: _Pod) -> Dict[str, Any]:
        return {
            "metadata": {"name": p.name, "namespace": p.namespace},
            "spec": {"containers": [{"image": p.image}]},
            "status": {"phase": p.phase, "ready": p.ready, "restarts": p.restarts, "reason": p.reason, "message": p.message},
        }

    def _telemetry(self) -> ClusterTelemetry:
        return ClusterTelemetry(
            scenario_id=self.task_id,
            step=self.step_n,
            time=_now(),
            pods=[self._pod_status(p) for (_, _), p in self._pods.items()],
            events=self._events[-15:],
            logs=[],
        )

    def _pod_status(self, p: _Pod) -> PodStatus:
        return PodStatus(
            name=p.name,
            namespace=p.namespace,
            phase=p.phase,  # type: ignore[arg-type]
            ready=p.ready,
            restarts=p.restarts,
            reason=p.reason,
            message=p.message,
            node="worker-0",
        )

    def _render(self, tool_output: Any = None, error: Optional[str] = None, extra: Optional[str] = None) -> str:
        t = self._telemetry()
        lines: List[str] = [f"task_id={t.scenario_id} step={t.step} done={self.done} score={self._get_safe_score():.2f}"]
        for p in t.pods:
            lines.append(f"pod {p.namespace}/{p.name} phase={p.phase} ready={p.ready} restarts={p.restarts} reason={p.reason}")
        if t.events:
            e = t.events[-1]
            lines.append(f"last_event: {e.reason} {e.message}")
        if extra:
            lines.append(extra)
        if error:
            lines.append(f"error: {error}")
        if tool_output is not None:
            try:
                lines.append("tool_output: " + json.dumps(tool_output)[:1200])
            except Exception:
                lines.append("tool_output: " + str(tool_output)[:1200])
        return "\n".join(lines)

    def _obs(self, text: str) -> KubeSREObservation:
        return KubeSREObservation(
            task_id=self.task_id,
            step=self.step_n,
            done=self.done,
            reward=float(self.reward),
            score=self._get_safe_score(),   # always strictly (0, 1)
            text=text,
            telemetry=self._telemetry(),
            metadata={"episode_id": self.episode_id},
        )