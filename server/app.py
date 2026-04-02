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
"""

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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)  # main()
