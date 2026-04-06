import os
import json
import subprocess
import inspect
import argparse
from typing import Generator, Tuple, Any, List, Dict, get_type_hints

from dotenv import load_dotenv
import httpx

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."

OLLAMA_BASE_URL = os.environ.get("AGENTO_OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_MODEL = os.environ.get("AGENTO_OLLAMA_MODEL")
OLLAMA_API_KEY = os.environ.get("AGENTO_OLLAMA_API_KEY")

if not OLLAMA_MODEL:
    raise ValueError("AGENTO_OLLAMA_MODEL must be set in environment")


def execute_bash(command: str) -> str:
    """Execute a bash command and return output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return f"[Exit {result.returncode}]\n{output}"


def read_file(path: str) -> str:
    """Read a file."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Write to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"


TOOLS = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
}


def _generate_tool_schema(func):
    """Generate JSON schema from function signature and type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = hints.get(name, str)
        json_type = type_map.get(param_type, "string")
        properties[name] = {"type": json_type}
        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


tools = [_generate_tool_schema(func) for func in TOOLS.values()]


def chat_stream(
    messages, tools=None, think=True
) -> Generator[Tuple[str, Any], None, None]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    request_body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "think": think,
    }

    if tools:
        request_body["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    with httpx.stream(
        "POST", url, json=request_body, headers=headers, timeout=300.0
    ) as response:
        response.raise_for_status()

        accumulated_thinking: List[str] = []
        accumulated_content: List[str] = []
        accumulated_tool_calls: List[Dict[str, Any]] = []

        for line in response.iter_lines():
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                yield ("error", f"Parse error: {e}")
                continue

            if chunk.get("error"):
                raise RuntimeError(f"Ollama error: {chunk['error']}")

            message = chunk.get("message", {})

            thinking = message.get("thinking")
            if thinking:
                accumulated_thinking.append(thinking)
                yield ("thinking", thinking)

            content = message.get("content")
            if content:
                accumulated_content.append(content)
                yield ("content", content)

            tool_calls = message.get("tool_calls")
            if tool_calls:
                accumulated_tool_calls.extend(tool_calls)
                yield ("tool_calls", tool_calls)

        final_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(accumulated_content),
        }
        if accumulated_thinking:
            final_message["thinking"] = "".join(accumulated_thinking)
        if accumulated_tool_calls:
            final_message["tool_calls"] = accumulated_tool_calls

        yield ("done", final_message)


def run_agent(user_message, messages=None, max_iterations=5):
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    else:
        messages.append({"role": "user", "content": user_message})

    for _ in range(max_iterations):
        thinking_parts = []
        content_parts = []
        tool_calls = []
        in_thinking = False
        in_content = False

        for chunk_type, chunk_data in chat_stream(messages, tools=tools, think=True):
            if chunk_type == "thinking":
                if not in_thinking:
                    print("[thinking] ", end="", flush=True)
                    in_thinking = True
                print(chunk_data, end="", flush=True)
                thinking_parts.append(chunk_data)
            elif chunk_type == "content":
                if in_thinking:
                    print("\n", end="", flush=True)
                    in_thinking = False
                if not in_content:
                    print("[response] ", end="", flush=True)
                    in_content = True
                print(chunk_data, end="", flush=True)
                content_parts.append(chunk_data)
            elif chunk_type == "tool_calls":
                tool_calls.extend(chunk_data)

        print()

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if thinking_parts:
            assistant_message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        messages.append(assistant_message)

        if not tool_calls:
            return assistant_message["content"]

        for tool_call in tool_calls:
            func_data = tool_call.get("function", {})
            name = func_data.get("name")
            args_str = func_data.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError as e:
                error_msg = f"Error: Invalid JSON in arguments for '{name}': {e}. Arguments received: {args_str!r}"
                print(f"[tool_error] {error_msg}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": error_msg,
                    }
                )
                continue

            print(f"[tool] {name}({args})")

            if name not in TOOLS:
                result = f"Error: Unknown tool '{name}'"
            else:
                sig = inspect.signature(TOOLS[name])
                try:
                    sig.bind(**args)
                    result = TOOLS[name](**args)
                except TypeError as e:
                    result = f"Error: Invalid arguments for '{name}': {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                }
            )

    return "Max iterations reached"


def interactive_loop():
    """Run an interactive chat session with persistent context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("Interactive mode started. Type /clear to clear session, Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/clear":
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
            ]
            print("Session cleared.\n")
            continue

        run_agent(user_input, messages=messages)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentO - AI assistant with tools")
    parser.add_argument(
        "-m", "--message", help="Send a single message and exit (non-interactive mode)"
    )

    args = parser.parse_args()

    if args.message:
        run_agent(args.message)
    else:
        interactive_loop()
