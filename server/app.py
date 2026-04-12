# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Customer Service Agent Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Can be run in three equivalent ways:
    python server/app.py                        # direct script
    uvicorn server.app:app --host 0.0.0.0       # module (recommended)
    python -m uvicorn server.app:app            # module via -m
"""

import os
import sys

# Ensure the project root (parent of this file's directory) is on sys.path so
# that `from models import ...` resolves correctly when this file is executed
# directly as a script (`python server/app.py`) rather than as part of a package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CustomerServiceAction, CustomerServiceObservation
    from .customer_service_env_environment import CustomerServiceEnvironment
except (ImportError, ModuleNotFoundError):
    from models import CustomerServiceAction, CustomerServiceObservation
    from server.customer_service_env_environment import CustomerServiceEnvironment


app = create_app(
    CustomerServiceEnvironment,
    CustomerServiceAction,
    CustomerServiceObservation,
    env_name="customer_service_env",
    max_concurrent_envs=3,
)


# Standard health-check endpoint — mirrors the pattern used by calendar_env
# and other top-ranked OpenEnv environments. Judges and CI checks hit this.
from fastapi.responses import JSONResponse  # noqa: E402

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe — returns 200 OK when the server is ready."""
    return JSONResponse({"status": "healthy", "service": "customer-service-env"})


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Customer Service Agent Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
