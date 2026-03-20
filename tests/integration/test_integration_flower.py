"""
tests/integration/test_infrastructure.py

Layer 1 — Infrastructure Tests
Verifies the docker stack is up and reachable before any federation runs.
Requires docker-compose stack to be running.

Run:
    uv run pytest tests/test_integration_flower.py -v
"""

import subprocess
import requests
import pytest
import grpc

pytestmark = pytest.mark.integration
# ── Configuration ─────────────────────────────────────────────────────────────

SUPERLINK_HOST      = "localhost"
SUPERLINK_FLEET_PORT = 9092   # Fleet API  — SuperNode → SuperLink
SUPERLINK_EXEC_PORT  = 9091   # Exec API   — flwr run  → SuperLink


# ── SuperLink Connectivity ────────────────────────────────────────────────────

class TestSuperLinkConnectivity:

    def test_superlink_exec_port_open(self):
        """Exec API must be reachable — flwr run communicates on this port."""
        channel = grpc.insecure_channel(f"{SUPERLINK_HOST}:{SUPERLINK_EXEC_PORT}")
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
            reachable = True
        except grpc.FutureTimeoutError:
            reachable = False
        finally:
            channel.close()
        assert reachable, (
            f"SuperLink Exec API not reachable at "
            f"{SUPERLINK_HOST}:{SUPERLINK_EXEC_PORT}"
        )

    def test_superlink_fleet_port_open(self):
        """Fleet API must be reachable — SuperNodes register on this port."""
        channel = grpc.insecure_channel(f"{SUPERLINK_HOST}:{SUPERLINK_FLEET_PORT}")
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
            reachable = True
        except grpc.FutureTimeoutError:
            reachable = False
        finally:
            channel.close()
        assert reachable, (
            f"SuperLink Fleet API not reachable at "
            f"{SUPERLINK_HOST}:{SUPERLINK_FLEET_PORT}"
        )


# ── SuperNode Registration ────────────────────────────────────────────────────

class TestSuperNodeRegistration:

    def test_supernodes_running(self):
        """At least 2 SuperNode containers must be running."""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=supernode", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        nodes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert len(nodes) >= 2, (
            f"Expected ≥2 SuperNode containers, found: {nodes}"
        )

    def test_supernodes_healthy(self):
        """All SuperNode containers must be in a running state."""
        result = subprocess.run(
            [
                "docker", "ps",
                "--filter", "name=supernode",
                "--format", "{{.Names}}\t{{.Status}}",
            ],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        assert lines, "No SuperNode containers found"
        for line in lines:
            name, status = line.split("\t", 1)
            assert "Up" in status, f"SuperNode '{name}' is not running: {status}"

    def test_superlink_running(self):
        """SuperLink container must be in a running state."""
        result = subprocess.run(
            [
                "docker", "ps",
                "--filter", "name=superlink",
                "--format", "{{.Names}}\t{{.Status}}",
            ],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        assert lines, "SuperLink container not found"
        for line in lines:
            name, status = line.split("\t", 1)
            assert "Up" in status, f"SuperLink '{name}' is not running: {status}"