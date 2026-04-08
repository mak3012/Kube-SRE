from __future__ import annotations

import os

import uvicorn
from openenv.core.env_server import create_fastapi_app
from fastapi.responses import HTMLResponse

# Note: The dots represent relative imports within the 'server' package
from .environment import KubeSREGymEnv
from .models import KubeSREObservation, KubeToolAction


APP_NAME = "kube-sre-gym"


def create_app():
    """
    OpenEnv app factory.

    This must return a FastAPI app created by openenv-core's create_fastapi_app
    so that automated trainers (and the hackathon runner) see the correct API.
    """

    app = create_fastapi_app(
        env=lambda: KubeSREGymEnv(seed=0),
        action_cls=KubeToolAction,
        observation_cls=KubeSREObservation,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Kube-SRE-Gym</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 28px; line-height: 1.45; }
      code { background: #f2f4f8; padding: 2px 6px; border-radius: 6px; }
      pre { background: #f6f8fa; padding: 12px; border-radius: 12px; overflow-x: auto; }
    </style>
  </head>
  <body>
    <h2>Kube-SRE-Gym</h2>
    <p>This is an <b>OpenEnv</b> environment server (openenv-core). Don’t use VSCode “Live Server” (port <code>5500</code>) for this.</p>
    <ul>
      <li>OpenAPI docs: <code>/docs</code></li>
      <li>WebSocket endpoint: <code>ws://127.0.0.1:&lt;PORT&gt;/ws</code></li>
    </ul>
    <h3>Correct local URLs</h3>
    <pre>http://127.0.0.1:8001/docs
ws://127.0.0.1:8001/ws</pre>
  </body>
</html>
""".strip()

    return app


# The global app object for the runner
app = create_app()


def main():
    """
    The entry point called by the 'server' script in pyproject.toml 
    and the Docker CMD runner.
    """
    # Use 0.0.0.0 for Docker/HuggingFace, default to 127.0.0.1 for local dev
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    
    # CRITICAL: The string path must be "server.app:app" to work as a package
    uvicorn.run("server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()