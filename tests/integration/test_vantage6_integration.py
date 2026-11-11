"""
Comprehensive Vantage6 integration testing.

This module tests the complete Vantage6 workflow:
1. Set up the vantage6 developer network (session-scoped)
2. Build the algorithm image (session-scoped)
3. Verify Docker containers are spawned correctly
4. Run tasks on the developer network
5. Assert results match expected central values
6. Clean up the network (at the session end)
"""

import pytest
import subprocess
import docker
import time
import re

import concurrent.futures

from typing import Dict, Any, List, Tuple


def cleanup_vantage6_network(
    network_info: Dict[str, Any],
    docker_client: docker.DockerClient,
    force_remove_existing: bool = False,
) -> bool:
    """Clean up network containers and resources."""
    try:
        # First, use CLI clean-up to gracefully stop the network
        def cli_cleanup():
            try:
                print("Attempting graceful CLI cleanup...")
                stop_result = subprocess.run(
                    ["v6", "dev", "stop-demo-network", "--name", "algorithm-ci-test"],
                    timeout=60,
                    capture_output=True,
                    text=True,
                )

                if stop_result.returncode == 0:
                    print("CLI stop was successful")

                time.sleep(2)  # Wait for graceful stop

                remove_result = subprocess.run(
                    ["v6", "dev", "remove-demo-network", "--name", "algorithm-ci-test"],
                    timeout=60,
                    capture_output=True,
                    text=True,
                )

                if remove_result.returncode == 0:
                    print("CLI remove successful")

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"CLI cleanup failed: {e}")

        # Try CLI clean-up first
        cli_cleanup()

        # Wait a moment for CLI clean-up to take effect
        time.sleep(3)

        containers_to_cleanup = []

        # Collect containers to clean up
        if network_info.get("created_containers"):
            containers_to_cleanup.extend(network_info["created_containers"])

        # If force_remove_existing is True, also find existing vantage6 containers
        if force_remove_existing:
            try:
                existing_containers = docker_client.containers.list(
                    all=True, filters={"name": "vantage6-algorithm-ci-test"}
                )
                containers_to_cleanup.extend([c.id for c in existing_containers])
            except Exception as e:
                print(f"Failed to list existing containers: {e}")

        # Remove duplicates
        containers_to_cleanup = list(set(containers_to_cleanup))

        def cleanup_container(container_id):
            """Clean up a single container."""
            try:
                container = docker_client.containers.get(container_id)
                container_name = container.name

                # Only stop if still running (CLI might have already stopped it)
                if container.status == "running":
                    container.stop(timeout=15)  # Give more time for a graceful stop
                    print(f"Stopped container: {container_name} ({container_id[:12]})")

                # Wait a moment for the container to fully stop
                time.sleep(1)

                # Remove the container
                container.remove(force=True)
                print(f"Removed container: {container_name} ({container_id[:12]})")
                return True

            except docker.errors.NotFound:
                # Container already removed (likely by CLI clean-up)
                return True
            except docker.errors.APIError as e:
                if "removal of container" in str(e) and "already in progress" in str(e):
                    # Another process is already removing this container, wait for it
                    for _ in range(10):  # Wait up to 10 seconds
                        try:
                            docker_client.containers.get(container_id)
                            time.sleep(1)
                        except docker.errors.NotFound:
                            print(f"Container {container_id[:12]} removed by another process")
                            return True
                    print(f"Container {container_id[:12]} removal timed out")
                    return False
                else:
                    print(f"Failed to cleanup container {container_id[:12]}: {e}")
                    return False
            except Exception as e:
                print(f"Failed to cleanup container {container_id[:12]}: {e}")
                return False

        # Clean up remaining containers in parallel for speed
        if containers_to_cleanup:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cleanup_container, cid) for cid in containers_to_cleanup]
                results = [f.result() for f in concurrent.futures.as_completed(futures, timeout=60)]  # noqa: F841

        return True

    except Exception as e:
        print(f"Network clean-up failed: {e}")
        return False


@pytest.mark.integration
class TestVantage6DeveloperNetwork:
    """Test the complete Vantage6 developer network workflow."""

    def test_vantage6_network_setup(self, vantage6_network_session, docker_client):
        """Test that the Vantage6 developer network is set up correctly."""
        assert vantage6_network_session["status"] == "running"

        # Check that Docker containers are running
        created_containers = vantage6_network_session["created_containers"]
        assert len(created_containers) >= 4, f"Expected at least 4 new containers, found {len(created_containers)}"

        # Categorise containers and check their expected states
        service_containers = []
        task_containers = []

        for container_id in created_containers:
            try:
                container = docker_client.containers.get(container_id)

                # Task/run containers are expected to exit after completing
                if "-run-" in container.name or "algorithm-store" in container.name:
                    task_containers.append(container)
                else:
                    # Service containers should stay running
                    service_containers.append(container)

            except docker.errors.NotFound:
                pytest.fail(f"Container {container_id[:12]} was not found")

        # Service containers (server, UI, nodes) should be running
        for container in service_containers:
            assert (
                container.status == "running"
            ), f"Service container {container.name} should be running: {container.status}"

        # Task containers can be in various states (running, exited)
        print(f"Found {len(service_containers)} service containers and {len(task_containers)} task containers")

        # Ensure we have at least the core service containers
        assert (
            len(service_containers) >= 3
        ), f"Expected at least 3 service containers (server, UI, node), found {len(service_containers)}"

    def _identify_container_types(self, docker_client, created_containers: List[str]) -> Dict[str, List]:
        """Identify server and node containers from created containers."""
        server_containers = []
        node_containers = []
        other_containers = []

        for container_id in created_containers:
            try:
                container = docker_client.containers.get(container_id)
                container_name = container.name.lower()

                if "server" in container_name or "api" in container_name:
                    server_containers.append(container)
                elif "node" in container_name and "run" not in container_name:
                    node_containers.append(container)
                else:
                    other_containers.append(container)

            except docker.errors.NotFound:
                continue

        return {
            "server": server_containers,
            "node": node_containers,
            "other": other_containers,
        }

    def _check_container_logs_for_connection(self, container, search_patterns: List[str]) -> Tuple[bool, List[str]]:
        """Check container logs for connection-related messages."""
        try:
            logs = container.logs(tail=-100).decode("utf-8", errors="ignore")
            found_patterns = []

            for pattern in search_patterns:
                if re.search(pattern, logs, re.IGNORECASE):
                    found_patterns.append(pattern)

            return len(found_patterns) > 0, found_patterns

        except Exception as e:
            print(f"Failed to get logs for {container.name}: {e}")
            return False, []

    def _check_network_connectivity(self, docker_client, node_container, server_container) -> bool:
        """Check if node container can reach server container."""
        try:
            # Get server container's IP address
            server_network = server_container.attrs["NetworkSettings"]["Networks"]
            server_ip = None

            for network_name, network_info in server_network.items():
                if network_info.get("IPAddress"):
                    server_ip = network_info["IPAddress"]
                    break

            if not server_ip:
                print(f"Could not determine server IP for {server_container.name}")
                return False

            # Try to ping server from node container (shorter timeout via ping options)
            exec_result = node_container.exec_run(f"ping -c 1 -W 2 {server_ip}")

            return exec_result.exit_code == 0

        except Exception as e:
            print(f"Network connectivity check failed: {e}")
            return False

    def test_node_server_connections(self, vantage6_network_session, docker_client):
        """Test that node containers are actually connecting to the server container."""
        created_containers = vantage6_network_session["created_containers"]
        containers = self._identify_container_types(docker_client, created_containers)

        server_containers = containers["server"]
        node_containers = containers["node"]

        assert len(server_containers) >= 1, "Expected at least one server container"
        assert len(node_containers) >= 1, "Expected at least one node container"

        server_container = server_containers[0]  # Use first server container

        print(f"Testing connections between {len(node_containers)} nodes and server {server_container.name}")

        # Patterns to look for in server logs indicating node connections
        server_connection_patterns = [
            r"node.*connected",
            r"authentication.*successful",
            r"websocket.*connected",
            r"client.*registered",
            r"handshake.*complete",
            r"node.*request",
            r"run.*completed",
            r"Sending result.*to the server",
        ]

        # Patterns to look for in node logs indicating server connection
        node_connection_patterns = [
            r"connected.*server",
            r"authentication.*successful",
            r"websocket.*established",
            r"logged.*in",
            r"connection.*established",
            r"Making request.*\|.*http.*/api/",
            r"Sending result.*to the server",
            r"node.*started",
            r"Started node",
        ]

        # Retry configuration
        max_retry_time = 120  # Maximum time to wait for connections (seconds)
        retry_interval = 15  # Time between retry attempts (seconds)
        start_time = time.time()

        print(f"Checking node connections with retry logic (max {max_retry_time}s, {retry_interval}s intervals)")

        while True:
            elapsed_time = time.time() - start_time

            # Check server logs for node connections
            server_has_connections, server_patterns = self._check_container_logs_for_connection(
                server_container, server_connection_patterns
            )

            if server_has_connections:
                print(f"Server shows connection indicators: {server_patterns}")

            # Check each node for connection to server
            connected_nodes = 0

            for node_container in node_containers:
                # Refresh container state to get latest logs
                try:
                    node_container.reload()
                except docker.errors.NotFound:
                    print(f"Node container {node_container.name} no longer exists")
                    continue

                # Check node logs for connection indicators
                node_connected, node_patterns = self._check_container_logs_for_connection(
                    node_container, node_connection_patterns
                )

                # Check network connectivity
                network_reachable = self._check_network_connectivity(docker_client, node_container, server_container)

                if node_connected:
                    print(f"Node {node_container.name} shows connection: {node_patterns}")
                    connected_nodes += 1
                elif network_reachable:
                    print(f"Node {node_container.name} can reach server but no clear connection logs")
                    connected_nodes += 1

            # Success condition: at least one node connected
            if connected_nodes > 0:
                print(
                    f"Connection verification successful: {connected_nodes}/{len(node_containers)} "
                    f"nodes connected (after {elapsed_time:.1f}s)"
                )
                break

            # Check if we've exceeded the maximum retry time
            if elapsed_time >= max_retry_time:
                print(f"Connection check timed out after {max_retry_time}s")
                break

            # Wait before next attempt
            print(
                f"No connections found yet (attempt at {elapsed_time:.1f}s), waiting {retry_interval}s before retry..."
            )
            time.sleep(retry_interval)

        # Final assertion - at least one node should be connected
        assert (
            connected_nodes > 0
        ), f"No nodes connected to server after {max_retry_time}s. Connected: {connected_nodes}/{len(node_containers)}"

    def test_container_health_status(self, vantage6_network_session, docker_client):
        """Test the health status of all running containers."""
        created_containers = vantage6_network_session["created_containers"]
        containers = self._identify_container_types(docker_client, created_containers)

        # Check health of server and node containers
        for container_type, container_list in [
            ("server", containers["server"]),
            ("node", containers["node"]),
        ]:
            for container in container_list:
                # Check if container is still running
                container.reload()  # Refresh container state
                assert (
                    container.status == "running"
                ), f"{container_type.title()} container {container.name} is not running: {container.status}"

                # Check health status if available
                health = container.attrs.get("State", {}).get("Health", {})
                if health:
                    health_status = health.get("Status", "unknown")
                    print(f"{container_type.title()} {container.name} health: {health_status}")

                    # If health check is configured, it should be healthy
                    if health_status in ["starting", "healthy"]:
                        continue  # These are acceptable states
                    elif health_status == "unhealthy":
                        pytest.fail(f"{container_type.title()} container {container.name} is unhealthy")


@pytest.mark.integration
class TestAlgorithmImage:
    """Test algorithm Docker image building and verification."""

    def test_algorithm_image_exists(self, algorithm_image, docker_client):
        """Test that the algorithm Docker image was built successfully."""
        expected_tag = f"{algorithm_image['pkg_name']}:ci-test"
        assert algorithm_image["tag"] == expected_tag
        assert algorithm_image["id"] is not None

        # Verify the image exists in Docker
        try:
            image = docker_client.images.get(algorithm_image["tag"])
            assert image.id == algorithm_image["id"]
            print(f"Algorithm image verified: {algorithm_image['tag']} ({algorithm_image['id'][:12]})")
        except docker.errors.ImageNotFound:
            pytest.fail(f"Built algorithm image {algorithm_image['tag']} not found in Docker")

    def test_algorithm_image_can_run(self, algorithm_image, docker_client):
        """Test that the algorithm image can be instantiated."""
        try:
            # Try to create a container from the image (don't run it)
            container = docker_client.containers.create(
                algorithm_image["tag"],
                command=[
                    "python",
                    "--version",
                ],  # Simple command to test if image works
            )

            # Clean up the test container
            container.remove()
            print("Algorithm image can be instantiated successfully")

        except docker.errors.APIError as e:
            pytest.fail(f"Failed to create container from algorithm image: {e}")
