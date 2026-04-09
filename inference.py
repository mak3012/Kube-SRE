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

# The validator provides OPENENV_ADDR. If running locally, default to 8000 (matching your Dockerfile)
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
                # Ping the health/docs endpoint
                res = await http.get(f"{ENV_BASE_URL}/docs", timeout=2.0)
                if res.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1) # Wait 1 second and retry
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

async def solve_task(task_id: str):
    print(f"[START] {task_id}")
    score = 0.0
    
    # Use a high timeout because LLMs can be slow
    async with httpx.AsyncClient(timeout=30.0) as http:
        try:
            # --- RESET ENVIRONMENT ---
            reset_res = await http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
            reset_res.raise_for_status()
            state = reset_res.json()
            
            # Cap steps to prevent infinite loops (Hackathon limit: 20 mins total)
            for step_num in range(15): 
                try:
                    # --- CALL LLM ---
                    # --- CALL LLM ---
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        response_format={ "type": "json_object" }, # FORCES JSON OUTPUT
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a Kubernetes SRE Agent. Return a JSON object representing the action to take. It must have a 'tool' field (e.g., 'get_pods', 'logs') and an 'args' dict."
                            },
                            {"role": "user", "content": f"Current State: {json.dumps(state)}"}
                        ]
                    )
                    raw_action = response.choices[0].message.content
                    
                    # --- PARSE ACTION ---
                    action_payload = clean_json_response(raw_action)
                    print(f"[STEP] {json.dumps(action_payload)}")
                    
                    # --- STEP ENVIRONMENT ---
                    step_res = await http.post(f"{ENV_BASE_URL}/step", json=action_payload)
                    step_res.raise_for_status()
                    step_data = step_res.json()
                    
                    # Extract variables based on standard OpenEnv schema
                    state = step_data.get("observation", step_data)
                    is_done = step_data.get("done", False)
                    score = float(step_data.get("reward", 0.0))
                    
                    if is_done:
                        break
                        
                # Catch Parsing Errors so the script doesn't crash
                except json.JSONDecodeError as je:
                    print(f"[STEP] JSON Parse Error: {je}")
                    break # End the task early with current score
                    
                # Catch Network/LLM Errors
                except Exception as e:
                    print(f"[STEP] Step Execution Error: {e}")
                    break
                    
        except httpx.RequestError as e:
            print(f"[STEP] Environment connection failed: {e}")
        except Exception as e:
            print(f"[STEP] Unhandled error during setup: {e}")
            
    # Always emit [END] even if the task failed midway!
    print(f"[END] {task_id} {score}")

async def main():
    print("Waiting for OpenEnv server to boot...")
    if not await wait_for_server():
        print("CRITICAL: Server did not start in time. Check Dockerfile and Port!")
        return
        
    # CRITICAL FIX: These must match the keys in SCENARIOS from environment.py
    tasks = ["ghost_image", "memory_leak", "cascading_mesh_failure"] 
    
    for t in tasks:
        await solve_task(t)

if __name__ == "__main__":
    asyncio.run(main())