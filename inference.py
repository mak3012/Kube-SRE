from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import websockets
from openai import OpenAI

# Read configuration from environment variables.
ENV_WS_URL = os.environ.get("ENV_WS_URL", "ws://127.0.0.1:8001/ws")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN = os.environ.get("HF_TOKEN")
# Fallback to OPENAI_API_KEY if HF_TOKEN isn't set.
API_KEY = HF_TOKEN or os.environ.get("OPENAI_API_KEY") 

SYSTEM_PROMPT = """You are an expert Staff SRE.
You are interacting with a simulated Kubernetes cluster through a strict, typed kubectl API.
Your goal: restore service health for the active incident scenario with minimal steps.

Output format:
- Return ONLY valid JSON with top-level key: "action"
- Example: {"action":{"kubectl":{"type":"kubectl","verb":"get","namespace":"prod","kind":"pod","output":"json"}}}
"""

def _client() -> OpenAI:
    if not API_KEY:
        raise ValueError("API Key is missing. Set HF_TOKEN or OPENAI_API_KEY.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def _openai_generate_json(client: OpenAI, system: str, user_json: str) -> str:
    # Combine prompts to keep a single user message.
    combined_prompt = f"{system}\n\nHere is the current cluster data:\n{user_json}"
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": combined_prompt},
        ],
        temperature=0.2,
    )
    
    text = resp.choices[0].message.content or "{}"
    
    # Strip markdown code fences if present.
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    return text.strip() or "{}"

async def _ws_send(ws, obj: Dict[str, Any]) -> None:
    await ws.send(json.dumps(obj))

async def _ws_recv(ws) -> Dict[str, Any]:
    raw = await ws.recv()
    return json.loads(raw)

def _make_action_from_text(text: str) -> Dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict) or "action" not in obj:
        raise ValueError("Model output must be JSON with top-level 'action'.")
    return obj["action"]

def _fmt_action(action: Dict[str, Any]) -> str:
    try:
        return json.dumps(action, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(action)

def _parse_state(msg: Dict[str, Any]) -> Dict[str, Any]:
    if "state" in msg:
        return msg["state"]
    if "data" in msg and isinstance(msg["data"], dict):
        if "state" in msg["data"]:
            return msg["data"]["state"]
        return msg["data"]
    print(f"\n[UNEXPECTED SERVER RESPONSE]: {msg}\n")
    raise ValueError("Missing 'state' in server response")

def _task_name_from_id(task_id: str) -> str:
    mapping = {
        "ghost_image": "The Ghost Image",
        "memory_leak": "The Memory Leak",
        "cascading_mesh_failure": "Cascading Mesh Failure",
    }
    return mapping.get(task_id, task_id)


async def solve_task(task_id: str, max_steps: int = 35, timeout_s: int = 600) -> Tuple[bool, int, float, List[float]]:
    """
    MUST emit stdout logs in this exact structure:
      task=<task_name> env=kube_sre_gym model=<model_name>
       step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    """

    task_name = _task_name_from_id(task_id)
    print(f"[START] task={task_name} env=kube_sre_gym model={MODEL_NAME}", flush=True)

    c: Optional[OpenAI] = None
    using_gemini = bool(GEMINI_API_KEY)
    if not using_gemini:
        if not OPENAI_API_KEY:
            # Fail fast but keep the hackathon-required stdout structure.
            print(" step=0 action={} reward=0.00 done=true error=missing_api_key", flush=True)
            print("  success=false steps=0 score=0.00 rewards=", flush=True)
            return False, 0, 0.0, []
        c = _client()
    rewards: List[float] = []
    t_start = time.time()

    async with websockets.connect(ENV_WS_URL, max_size=8 * 1024 * 1024) as ws:
        await _ws_send(ws, {"type": "reset", "scenario_id": task_id})
        msg = await _ws_recv(ws)
        state = _parse_state(msg)

        done = bool(state.get("terminated") or state.get("truncated"))
        last_error: Optional[str] = None

        for n in range(1, max_steps + 1):
            if time.time() - t_start > timeout_s:
                last_error = "timeout"
                break
            if done:
                break
                
            # Add a 13-second pause to prevent hitting Google's 5 requests/minute limit
            await asyncio.sleep(13)

            # Keep the prompt compact for latency/cost.
            obs = state.get("observation", {})
            pods = obs.get("pods", [])
            events = obs.get("events", [])
            last_event = events[-1] if events else {}
            prompt = {
                "task_id": task_id,
                "step": obs.get("step"),
                "pods": pods,
                "last_event": last_event,
                "last_kubectl_output": state.get("info", {}).get("kubectl_output", {}),
            }

            action: Dict[str, Any]
            try:
                user_payload = json.dumps(prompt)
                if using_gemini:
                    content = _gemini_generate_json(SYSTEM_PROMPT, user_payload)
                else:
                    assert c is not None
                    content = _openai_generate_json(c, SYSTEM_PROMPT, user_payload)
                action = _make_action_from_text(content)
                last_error = None
            except Exception as e:
                # Force the script to print the actual complaint from Google
                print(f"\n[DEBUG GEMINI ERROR]: {e}\n")
                
                # Deterministic fallback action (valid) if the model output fails.
                last_error = f"model_error:{e.__class__.__name__}"
                action = {
                    "kubectl": {
                        "type": "kubectl",
                        "verb": "get",
                        "namespace": "prod",
                        "kind": "pod",
                        "output": "json",
                    }
                }

            await _ws_send(ws, {"type": "step", "action": action})
            msg = await _ws_recv(ws)
            state = _parse_state(msg)

            r = float(state.get("reward", 0.0))
            rewards.append(r)
            done = bool(state.get("terminated") or state.get("truncated"))

            reward_str = f"{r:.2f}"
            done_str = "true" if done else "false"
            err_str = "null" if last_error is None else last_error
            print(f"[STEP] step={n} action={_fmt_action(action)} reward={reward_str} done={done_str} error={err_str}", flush=True)

        success = bool(state.get("terminated")) and state.get("info", {}).get("final") == "success"
        steps = int(state.get("observation", {}).get("step", len(rewards)))
        score = float(state.get("score", 0.0))
        rewards_str = ",".join([f"{x:.2f}" for x in rewards])
        print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

        return success, steps, score, rewards


async def main() -> None:
    # Keep runtime under 20 minutes: 3 tasks * (<=35 steps) with compact prompts.
    t0 = time.time()
    for task_id in ("ghost_image", "memory_leak", "cascading_mesh_failure"):
        await solve_task(task_id)
    _ = time.time() - t0


if __name__ == "__main__":
    asyncio.run(main())

