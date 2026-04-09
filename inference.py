import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
import traceback

# --- 1. MANDATORY HACKATHON VARIABLES ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "dummy-token-if-local")

# The validator provides OPENENV_ADDR. Default to 8000
ENV_BASE_URL = os.environ.get("OPENENV_ADDR", "http://localhost:8000").rstrip("/")

# --- 2. INITIALIZE OPENAI CLIENT ---
client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

async def wait_for_server():
    """Prevents ConnectionRefusedError by waiting for the FastAPI server to boot."""
    async with httpx.AsyncClient() as http:
        for attempt in range(15):
            try:
                res = await http.get(f"{ENV_BASE_URL}/docs", timeout=2.0)
                if res.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False

def clean_json_response(raw_text: str) -> dict:
    """Prevents JSONDecodeError by stripping Markdown formatting."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def _safe_score(value) -> float:
    """
    Guarantees score is strictly within (0, 1) — never 0.0, never 1.0.
    Called at every point a score value is read or returned.
    """
    try:
        s = float(value)
    except (TypeError, ValueError):
        s = 0.01
    # Hard clamp: strictly (0, 1)
    return max(0.01, min(0.99, s))

async def solve_task(task_id: str) -> float:
    """
    Runs one episode for the given task_id and returns a score
    strictly within (0, 1).
    """
    print(f"[START] {task_id}")
    score = 0.01  # Safe default — strictly within (0, 1)

    async with httpx.AsyncClient(timeout=30.0) as http:
        try:
            # --- RESET ENVIRONMENT ---
            reset_res = await http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
            reset_res.raise_for_status()
            state = reset_res.json()

            for step_num in range(15):
                try:
                    # --- CALL LLM ---
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        response_format={"type": "json_object"},  # Forces JSON output
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a Kubernetes SRE Agent. "
                                    "Return a JSON object representing the action to take. "
                                    "It must have a 'tool' field (e.g., 'get_pods', 'logs') "
                                    "and an 'args' dict."
                                ),
                            },
                            {"role": "user", "content": f"Current State: {json.dumps(state)}"},
                        ],
                    )
                    raw_action = response.choices[0].message.content

                    # --- PARSE ACTION ---
                    action_payload = clean_json_response(raw_action)
                    print(f"[STEP {step_num}] {json.dumps(action_payload)}")

                    # --- STEP ENVIRONMENT ---
                    step_res = await http.post(f"{ENV_BASE_URL}/step", json=action_payload)
                    step_res.raise_for_status()
                    step_data = step_res.json()

                    # --- EXTRACT VARIABLES ---
                    state = step_data.get("observation", step_data)
                    is_done = step_data.get("done", False)

                    # Extract and clamp score at every read — never trust raw env value
                    raw = None
                    if isinstance(state, dict):
                        raw = state.get("score")
                    if raw is None:
                        raw = step_data.get("score")
                    score = _safe_score(raw)

                    if is_done:
                        break

                # Catch Parsing Errors
                except json.JSONDecodeError as je:
                    print(f"[STEP {step_num}] JSON Parse Error: {je}")
                    break

                # Catch Network/LLM Errors
                except Exception as e:
                    print(f"[STEP {step_num}] Step Execution Error: {e}")
                    traceback.print_exc()
                    break

        except httpx.RequestError as e:
            print(f"[TASK] Environment connection failed: {e}")
        except Exception as e:
            print(f"[TASK] Unhandled error during setup: {e}")
            traceback.print_exc()

    # --- FINAL SAFETY CLAMP ---
    # Applied once more before returning — score is always strictly (0, 1)
    final_score = _safe_score(score)
    print(f"[END] {task_id} → score={final_score}")
    return final_score  # ← CRITICAL: must return the score


async def main():
    print("Waiting for OpenEnv server to boot...")
    if not await wait_for_server():
        print("CRITICAL: Server did not start in time. Check Dockerfile and Port!")
        return {}

    # Matches environment.py SCENARIOS perfectly
    tasks = ["ghost_image", "memory_leak", "cascading_mesh_failure"]

    scores = {}
    for task_id in tasks:
        task_score = await solve_task(task_id)
        scores[task_id] = task_score  # guaranteed strictly within (0, 1)

    print(f"\n[SUMMARY] Final scores: {scores}")
    return scores  # ← return the full dict so callers/validators can read it


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, indent=2))