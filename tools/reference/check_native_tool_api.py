#!/usr/bin/env python3
"""Offline native tool-call API validation; never imported by the runtime."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    run = Path(sys.argv[1])
    port = int(os.environ.get("DSV41_SMOKE_PORT", "18083"))
    timeout = float(os.environ.get("DSV41_CHECK_TIMEOUT", "1800"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))

    env = dict(os.environ, DSV41_NATIVE_MODEL="1", DSV41_BIND=f"127.0.0.1:{port}")
    base = f"http://127.0.0.1:{port}"
    model = env.get("DSV41_MODEL", "DeepSeek-V4.1-Flash")
    max_tokens = int(os.environ.get("DSV41_TOOL_MAX_TOKENS", "128"))
    assert 16 <= max_tokens <= 256
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call get_weather for Tokyo. Do not answer without using the tool.",
            }
        ],
        "tools": [tool],
        "tool_choice": "required",
        "thinking": {"type": "disabled"},
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
    }
    (run / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(payload, deadline=timeout):
        encoded = json.dumps(payload).encode()
        http_request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(http_request, timeout=deadline) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def save(name, status, body):
        (run / f"{name}.json").write_text(
            json.dumps({"status": status, "body": body}, ensure_ascii=False, indent=2)
            + "\n"
        )

    def validate_calls(message):
        calls = message.get("tool_calls")
        assert calls and message.get("role") == "assistant", message
        assert len({item["id"] for item in calls}) == len(calls)
        functions = []
        for item in calls:
            assert item["type"] == "function"
            function = item["function"]
            assert function["name"] == "get_weather", function
            arguments = json.loads(function["arguments"])
            assert "tokyo" in str(arguments.get("location", "")).lower(), arguments
            functions.append((function["name"], arguments))
        return functions

    with (run / "server.log").open("w") as log:
        process = subprocess.Popen(
            ["server/target/debug/dsv41-developer-server"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {process.returncode}")
                try:
                    with opener.open(base + "/health", timeout=2) as response:
                        health = json.load(response)
                        if response.status == 200 and health["runtime"] == "native-model":
                            save("health", response.status, health)
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("model startup timed out")

            status, body = call(request)
            save("completion", status, body)
            assert status == 200, body
            choice = body["choices"][0]
            assert choice["finish_reason"] == "tool_calls", choice
            expected_functions = validate_calls(choice["message"])
            usage = body["usage"]
            assert usage["prompt_tokens"] > 0
            assert 0 < usage["completion_tokens"] <= max_tokens
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

            stream_request = dict(
                request, stream=True, stream_options={"include_usage": True}
            )
            http_request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(stream_request).encode(),
                headers={"Content-Type": "application/json"},
            )
            events = []
            with opener.open(http_request, timeout=timeout) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "text/event-stream"
                with (run / "completion-stream.sse").open("w") as record:
                    for raw in response:
                        line = raw.decode()
                        record.write(line)
                        record.flush()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        assert "error" not in event, event
                        events.append(event)
                    else:
                        raise AssertionError("SSE ended without [DONE]")

            streamed = {}
            finishes = []
            for event in events:
                for stream_choice in event.get("choices", []):
                    if stream_choice.get("finish_reason"):
                        finishes.append(stream_choice["finish_reason"])
                    for delta in stream_choice.get("delta", {}).get("tool_calls", []):
                        index = delta["index"]
                        item = streamed.setdefault(
                            index, {"id": None, "type": None, "name": None, "arguments": ""}
                        )
                        item["id"] = delta.get("id", item["id"])
                        item["type"] = delta.get("type", item["type"])
                        function = delta.get("function", {})
                        item["name"] = function.get("name", item["name"])
                        item["arguments"] += function.get("arguments", "")
            streamed_message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": item["id"],
                        "type": item["type"],
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }
                    for _, item in sorted(streamed.items())
                ],
            }
            assert finishes == ["tool_calls"]
            assert validate_calls(streamed_message) == expected_functions
            usage_events = [event["usage"] for event in events if event.get("usage")]
            assert len(usage_events) == 1 and usage_events[0] == usage
            assert events[-1]["choices"] == []

            followup = {
                "model": model,
                "messages": request["messages"]
                + [choice["message"]]
                + [
                    {
                        "role": "tool",
                        "tool_call_id": choice["message"]["tool_calls"][0]["id"],
                        "content": '{"temperature_c":22,"condition":"sunny"}',
                    }
                ],
                "tools": [tool],
                "tool_choice": "none",
                "thinking": {"type": "disabled"},
                "max_tokens": 16,
                "temperature": 0,
                "seed": 0,
            }
            (run / "followup-request.json").write_text(
                json.dumps(followup, ensure_ascii=False, indent=2) + "\n"
            )
            followup_status, followup_body = call(followup)
            save("followup", followup_status, followup_body)
            assert followup_status == 200, followup_body
            followup_choice = followup_body["choices"][0]
            assert followup_choice["message"].get("content"), followup_choice
            assert not followup_choice["message"].get("tool_calls"), followup_choice

            (run / "result.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "scope": "native tool request rendering, nonstream/SSE parsing parity, and historical tool result continuation; not 256K qualification",
                        "tool_calls": len(expected_functions),
                        "completion_tokens": usage["completion_tokens"],
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                "PASS: native tool rendering, HTTP/SSE tool-call parity and tool-result continuation",
                flush=True,
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
