"""Sample A2A client demonstrating discovery, send, polling, and cancel operations.

Usage:
    pip install gradient-adk[a2a] httpx
    python examples/a2a/client.py
"""

import asyncio
import httpx


async def discover_agent(base_url: str) -> dict:
    """Discover agent capabilities via AgentCard."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/.well-known/agent-card.json")
        response.raise_for_status()
        agent_card = response.json()
        print(f"Discovered agent: {agent_card['name']}")
        print(f"Transport URL: {agent_card['url']}")
        return agent_card


async def send_message(rpc_url: str, message_text: str) -> dict:
    """Send a message to the agent."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": message_text}],
                        "message_id": "msg-1",
                        "kind": "message",
                    }
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise Exception(f"Error: {result['error']}")
        return result["result"]


async def get_task(rpc_url: str, task_id: str) -> dict:
    """Poll task status."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise Exception(f"Error: {result['error']}")
        return result["result"]


async def cancel_task(rpc_url: str, task_id: str) -> dict:
    """Cancel a task."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": "3",
                "method": "tasks/cancel",
                "params": {"id": task_id},
            },
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise Exception(f"Error: {result['error']}")
        return result["result"]


async def main():
    """Demonstrate A2A client operations."""
    base_url = "http://localhost:8000"
    rpc_url = f"{base_url}/"

    print("=== A2A Client Demo ===\n")

    # 1. Discover agent
    print("1. Discovering agent...")
    agent_card = await discover_agent(base_url)
    print()

    # 2. Send message
    print("2. Sending message...")
    send_result = await send_message(rpc_url, "Hello, A2A!")
    task_id = send_result["id"]
    print(f"   Task ID: {task_id}")
    print(f"   Status: {send_result['status']['state']}")
    print()

    # 3. Poll task status
    print("3. Polling task status...")
    task = await get_task(rpc_url, task_id)
    print(f"   Task ID: {task['id']}")
    print(f"   Status: {task['status']['state']}")
    if task.get("response"):
        parts = task["response"].get("parts", [])
        if parts:
            print(f"   Response: {parts[0].get('text', 'N/A')}")
    print()

    # 4. Cancel task (example)
    print("4. Canceling task (example)...")
    send_result2 = await send_message(rpc_url, "This will be canceled")
    task_id2 = send_result2["id"]
    cancel_result = await cancel_task(rpc_url, task_id2)
    print(f"   Task ID: {cancel_result['id']}")
    print(f"   Status: {cancel_result['status']['state']}")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
