from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, StringConstraints, model_validator


# ----------------------------
# Action space (typed kubectl)
# ----------------------------

# Strict-ish resource identifiers to avoid injection-like payloads.
# We intentionally disallow whitespace and shell metacharacters.
K8sName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    ),
]

K8sNamespace = K8sName

ContainerName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=63,
        pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    ),
]

LabelSelector = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        # Basic label selector subset: key=value[,key=value...]
        pattern=r"^[A-Za-z0-9_.-]+=[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+=[A-Za-z0-9_.-]+)*$",
    ),
]


class K8sKind(str, Enum):
    pod = "pod"
    deployment = "deployment"
    service = "service"
    virtualservice = "virtualservice"
    namespace = "namespace"


class KubectlVerb(str, Enum):
    get = "get"
    describe = "describe"
    logs = "logs"
    patch = "patch"
    delete = "delete"
    edit = "edit"


class BaseAction(BaseModel):
    type: Literal["kubectl"] = "kubectl"
    verb: KubectlVerb
    namespace: Optional[K8sNamespace] = None
    request_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied correlation id for tracing.",
        max_length=128,
    )


class KubectlGet(BaseAction):
    verb: Literal[KubectlVerb.get] = KubectlVerb.get
    kind: K8sKind
    name: Optional[K8sName] = None
    selector: Optional[LabelSelector] = None
    output: Literal["json", "name", "wide"] = "json"

    @model_validator(mode="after")
    def _validate_name_or_selector(self) -> "KubectlGet":
        if self.name and self.selector:
            raise ValueError("Provide either name or selector, not both.")
        return self


class KubectlDescribe(BaseAction):
    verb: Literal[KubectlVerb.describe] = KubectlVerb.describe
    kind: K8sKind
    name: K8sName


class KubectlLogs(BaseAction):
    verb: Literal[KubectlVerb.logs] = KubectlVerb.logs
    pod: K8sName
    container: Optional[ContainerName] = None
    tail_lines: int = Field(default=200, ge=1, le=2000)


class PatchType(str, Enum):
    merge = "merge"
    json = "json"


class KubectlPatch(BaseAction):
    verb: Literal[KubectlVerb.patch] = KubectlVerb.patch
    kind: K8sKind
    name: K8sName
    patch_type: PatchType = PatchType.merge
    patch: Dict[str, Any] = Field(default_factory=dict)


class KubectlDelete(BaseAction):
    verb: Literal[KubectlVerb.delete] = KubectlVerb.delete
    kind: K8sKind
    name: K8sName
    grace_seconds: int = Field(default=0, ge=0, le=60)


class KubectlEdit(BaseAction):
    verb: Literal[KubectlVerb.edit] = KubectlVerb.edit
    kind: K8sKind
    name: K8sName
    # We model edit as "replace these fields" to keep it safe/deterministic.
    spec_patch: Dict[str, Any] = Field(default_factory=dict)


KubectlAction = Union[
    KubectlGet,
    KubectlDescribe,
    KubectlLogs,
    KubectlPatch,
    KubectlDelete,
    KubectlEdit,
]


class OpenEnvAction(BaseModel):
    """
    OpenEnv action wrapper.
    For this environment, the only action family is typed kubectl commands.
    """

    kubectl: KubectlAction


# ----------------------------
# Observation space (telemetry)
# ----------------------------


class EventSnippet(BaseModel):
    ts: float
    involved_object: str
    reason: str
    message: str


class PodStatus(BaseModel):
    name: K8sName
    namespace: K8sNamespace
    phase: Literal["Pending", "Running", "Succeeded", "Failed", "Unknown"]
    ready: bool
    restarts: int = Field(ge=0)
    reason: Optional[str] = None
    message: Optional[str] = None
    node: Optional[str] = None


class LogBuffer(BaseModel):
    pod: K8sName
    container: Optional[ContainerName] = None
    tail: List[str] = Field(default_factory=list)


class ClusterTelemetry(BaseModel):
    scenario_id: str
    step: int
    time: float
    pods: List[PodStatus] = Field(default_factory=list)
    events: List[EventSnippet] = Field(default_factory=list)
    logs: List[LogBuffer] = Field(default_factory=list)


class OpenEnvState(BaseModel):
    """
    OpenEnv state returned by /reset and /step.
    """

    observation: ClusterTelemetry
    reward: float
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic grader score in [0,1] for the current scenario.",
    )
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class OpenEnvResetRequest(BaseModel):
    scenario_id: Optional[str] = Field(
        default=None,
        description="If omitted, server chooses next scenario.",
    )
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


class OpenEnvStepRequest(BaseModel):
    action: OpenEnvAction


class WSClientMessage(BaseModel):
    """
    WebSocket message format: one of reset/step/state.
    """

    op: Literal["reset", "step", "state", "ping"]
    reset: Optional[OpenEnvResetRequest] = None
    step: Optional[OpenEnvStepRequest] = None


class WSServerMessage(BaseModel):
    op: Literal["reset_ok", "step_ok", "state_ok", "error", "pong"]
    state: Optional[OpenEnvState] = None
    error: Optional[str] = None
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenEnv-core compatible Action/Observation (for create_fastapi_app + GRPO)
# ---------------------------------------------------------------------------


class ToolName(str, Enum):
    get_pods = "get_pods"
    describe_pod = "describe_pod"
    logs = "logs"
    patch_deployment = "patch_deployment"
    patch_virtualservice = "patch_virtualservice"


class GetPodsArgs(BaseModel):
    namespace: Optional[K8sNamespace] = Field(default="prod")


class DescribePodArgs(BaseModel):
    name: K8sName
    namespace: Optional[K8sNamespace] = Field(default="prod")


class LogsArgs(BaseModel):
    pod: K8sName
    namespace: Optional[K8sNamespace] = Field(default="prod")
    tail_lines: int = Field(default=200, ge=1, le=2000)


class PatchDeploymentArgs(BaseModel):
    name: K8sName
    namespace: Optional[K8sNamespace] = Field(default="prod")
    patch: Dict[str, Any] = Field(default_factory=dict)


class PatchVirtualServiceArgs(BaseModel):
    name: K8sName
    namespace: Optional[K8sNamespace] = Field(default="prod")
    patch: Dict[str, Any] = Field(default_factory=dict)


class KubeToolAction(Action):
    """
    Action format used by openenv-core HTTP/WS server.

    The GRPOTrainer will discover callable tools via public methods on the env,
    but we also keep an explicit action schema for deterministic routing when
    used through the env server.
    """

    tool: ToolName
    args: Union[
        GetPodsArgs,
        DescribePodArgs,
        LogsArgs,
        PatchDeploymentArgs,
        PatchVirtualServiceArgs,
    ]


class KubeSREObservation(Observation):
    """
    Observation returned by the openenv-core Environment.
    """

    task_id: str = Field(description="Scenario id (task identifier).")
    step: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    text: str = Field(description="LLM-readable observation string.")
    telemetry: ClusterTelemetry