"""Check the installed text generation protocols against a live direct or managed model."""

import argparse
import base64
import http.client
import json
import os
import struct
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit


def exchange(url, path, body, headers, *, cancel=False):
    parsed = urlsplit(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)
    try:
        conn.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json", **headers}
        )
        response = conn.getresponse()
        if response.status != 200 or "text/event-stream" not in response.getheader(
            "Content-Type", ""
        ):
            return response.status, json.loads(response.read())
        events = []
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            events.append("[DONE]" if value == b"[DONE]" else json.loads(value))
            if cancel and any(
                isinstance(e, dict) and e.get("type") == "response.output_text.delta"
                for e in events
            ):
                break
        return response.status, events
    finally:
        conn.close()


def check(url, model, key, report, managed):
    auth = {"Authorization": f"Bearer {key}"}
    anth = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    baseline = {"model": model, "temperature": 0, "max_tokens": 16}
    prompt = "What is the capital of France? Answer only with the city name."
    messages = {**baseline, "messages": [{"role": "user", "content": prompt}]}
    responses = {"model": model, "input": prompt, "temperature": 0, "max_output_tokens": 16}
    rows = {}

    def call(tag, path, body, headers, want=200, **kw):
        status, response = exchange(url, path, body, headers, **kw)
        rows[tag] = {"status": status, "response": response}
        report.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        assert status == want, (tag, status, response)
        return response

    call("missing-auth", "/v1/responses", responses, {}, 401)
    err = call("bad-key", "/v1/messages", messages, {**anth, "x-api-key": "invalid"}, 401)
    assert err["error"]["type"] == "authentication_error"
    call("bad-version", "/v1/messages", messages, {**anth, "anthropic-version": "invalid"}, 400)
    call("stored", "/v1/responses", {**responses, "store": True}, auth, 400)
    call(
        "unknown-model",
        "/v1/messages",
        {**messages, "model": "unknown-protocol-test"},
        anth,
        404 if managed else 400,
    )
    count = call(
        "count",
        "/v1/messages/count_tokens",
        {"model": model, "messages": messages["messages"]},
        anth,
    )["input_tokens"]
    native = call("messages-json", "/v1/messages", messages, anth)
    assert native["model"] == model and native["stop_reason"] == "end_turn"
    assert native["usage"]["input_tokens"] + native["usage"]["cache_read_input_tokens"] == count
    answer = "".join(b["text"] for b in native["content"])
    assert answer.strip().rstrip(".") == "Paris", answer
    stream = call("messages-sse", "/v1/messages", {**messages, "stream": True}, anth)
    assert stream[0]["type"] == "message_start" and stream[-1]["type"] == "message_stop"
    assert (
        "".join(
            e["delta"]["text"] for e in stream if e.get("delta", {}).get("type") == "text_delta"
        )
        == answer
    )
    assert stream[-2]["usage"]["output_tokens"] == native["usage"]["output_tokens"]
    stopped = call(
        "stop", "/v1/messages", {**messages, "stop_sequences": ["arsenic", "aris"]}, anth
    )
    assert (stopped["stop_reason"], stopped["stop_sequence"]) == ("stop_sequence", "aris"), stopped
    native = call("responses-json", "/v1/responses", responses, auth)
    assert native["model"] == model and native["output"][0]["content"][0]["text"] == answer
    stream = call("responses-sse", "/v1/responses", {**responses, "stream": True}, auth)
    assert [e["sequence_number"] for e in stream] == list(range(len(stream)))
    assert (
        "".join(e["delta"] for e in stream if e["type"] == "response.output_text.delta") == answer
    )
    assert (
        stream[-1]["type"] == "response.completed"
        and stream[-1]["response"]["usage"] == native["usage"]
    )
    chat = call(
        "chat-regression",
        "/v1/chat/completions",
        {**messages, "stream": True, "stream_options": {"include_usage": True}},
        auth,
    )
    assert chat[-1] == "[DONE]" and chat[-2]["usage"]["prompt_tokens"] == count
    assert "".join(c["delta"].get("content", "") for e in chat[:-1] for c in e["choices"]) == answer
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    query = "Call inspect with the label BANANA. Use the tool and do not answer in text."
    for api in ("messages", "responses"):
        msg = api == "messages"
        h = anth if msg else auth
        body = dict(
            model=model,
            stream=True,
            temperature=0,
            tools=[
                {"name": "inspect", "input_schema": schema}
                if msg
                else {"type": "function", "name": "inspect", "parameters": schema}
            ],
        )
        body.update(
            {
                "messages": [{"role": "user", "content": query}],
                "tool_choice": {"type": "tool", "name": "inspect"},
                "max_tokens": 64,
            }
            if msg
            else {
                "input": query,
                "tool_choice": {"type": "function", "name": "inspect"},
                "max_output_tokens": 64,
            }
        )
        stream = call(api + "-tool", "/v1/" + api, body, h)
        if msg:
            start = next(
                e["content_block"]
                for e in stream
                if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"
            )
            value = "".join(
                e["delta"]["partial_json"]
                for e in stream
                if e.get("delta", {}).get("type") == "input_json_delta"
            )
            assert (
                start["name"] == "inspect"
                and json.loads(value)["label"] == "BANANA"
                and stream[-2]["delta"]["stop_reason"] == "tool_use"
            )
        else:
            item = next(
                e["item"]
                for e in stream
                if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
            )
            value = "".join(
                e["delta"] for e in stream if e["type"] == "response.function_call_arguments.delta"
            )
            assert (
                item["arguments"] == value
                and json.loads(value)["label"] == "BANANA"
                and item["id"] != item["call_id"]
            )
        full = call(api + "-tool-json", "/v1/" + api, {**body, "stream": False}, h)
        if msg:
            block = next(b for b in full["content"] if b["type"] == "tool_use")
            assert (block["name"], block["input"]) == (start["name"], json.loads(value))
            assert full["stop_reason"] == "tool_use" and full["usage"] == stream[-2]["usage"]
        else:
            block = next(b for b in full["output"] if b["type"] == "function_call")
            assert (block["name"], block["arguments"]) == (item["name"], value)
            assert (
                full["status"] == "completed" and full["usage"] == stream[-1]["response"]["usage"]
            )
        body["max_tokens" if msg else "max_output_tokens"] = 1
        cut = call(api + "-cut", "/v1/" + api, body, h)
        if msg:
            assert cut[-2]["delta"]["stop_reason"] == "max_tokens" and not any(
                e.get("content_block", {}).get("type") == "tool_use" for e in cut
            )
        else:
            assert cut[-1]["type"] == "response.incomplete" and not any(
                e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
                for e in cut
            )

    def png_chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    side = 112
    image = (
        bytes.fromhex("89504e470d0a1a0a")
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress((bytes(1) + bytes([255, 0, 0]) * side) * side))
        + png_chunk(b"IEND", b"")
    )
    image = base64.b64encode(image).decode()
    for api in ("responses", "messages"):
        msg = api == "messages"
        headers = anth if msg else auth
        parts = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image},
            }
            if msg
            else {"type": "input_image", "image_url": "data:image/png;base64," + image},
            {
                "type": "text" if msg else "input_text",
                "text": "What color fills the image? Answer only with the color name.",
            },
        ]
        key = "messages" if msg else "input"
        body = {**(messages if msg else responses), key: [{"role": "user", "content": parts}]}
        image_result = call(api + "-image", "/v1/" + api, body, headers)
        blocks = image_result["content"] if msg else image_result["output"][0]["content"]
        assert "red" in "".join(b["text"] for b in blocks).lower()
        if msg:
            measured = call("count-image", "/v1/messages/count_tokens", body, headers)
            assert measured["input_tokens"] == image_result["usage"]["input_tokens"]
        # Replay two calls to the same function with intervening assistant text.
        # Return the second result first. Include the follow-up in the same turn.
        query = {"role": "user", "content": "Look up the two labels."}
        question = "Reply only with the value returned by the FIRST call."
        names = ("a", "b")
        if msg:
            blocks = []
            for name in names:
                blocks.extend(
                    [
                        {"type": "text", "text": "Looking up " + name},
                        {
                            "type": "tool_use",
                            "id": name,
                            "name": "inspect",
                            "input": {"label": name},
                        },
                    ]
                )
            blocks.append({"type": "text", "text": "Waiting for both results."})
            history = [
                query,
                {"role": "assistant", "content": blocks},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "b", "content": "LEMON"},
                        {"type": "tool_result", "tool_use_id": "a", "content": "MANGO"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
        else:
            history = [query]
            for name in names:
                history.extend(
                    [
                        {"role": "assistant", "content": "Looking up " + name},
                        {
                            "type": "function_call",
                            "name": "inspect",
                            "arguments": json.dumps({"label": name}),
                            "call_id": name,
                        },
                    ]
                )
            history.extend(
                [
                    {"role": "assistant", "content": "Waiting for both results."},
                    {"type": "function_call_output", "call_id": "b", "output": "LEMON"},
                    {"type": "function_call_output", "call_id": "a", "output": "MANGO"},
                    {"role": "user", "content": question},
                ]
            )
        body[key] = history
        body["tools"] = [
            {"name": "inspect", "input_schema": schema}
            if msg
            else {"type": "function", "name": "inspect", "parameters": schema}
        ]
        body["tool_choice"] = {"type": "none"} if msg else "none"
        result = call(api + "-ordered-history", "/v1/" + api, body, headers)
        blocks = result["content"] if msg else result["output"][0]["content"]
        assert "MANGO" in "".join(b["text"] for b in blocks).upper(), blocks
        if msg:
            assert (
                call("count-history", "/v1/messages/count_tokens", body, headers)["input_tokens"]
                == result["usage"]["input_tokens"]
            )
    call(
        "cancel",
        "/v1/responses",
        {
            **responses,
            "stream": True,
            "input": "List the integers from 1 to 1000 with a comma between each.",
            "max_output_tokens": 1024,
        },
        auth,
        cancel=True,
    )
    if not managed:
        parsed = urlsplit(url)
        for _ in range(100):
            c = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
            try:
                c.request("GET", "/health", headers=auth)
                active = json.loads(c.getresponse().read())["active_requests"]
            finally:
                c.close()
            if not active:
                break
            time.sleep(0.05)
        assert not active
    assert (
        call("after-cancel", "/v1/responses", responses, auth)["output"][0]["content"][0]["text"]
        == answer
    )
    print(f"{len(rows)} HTTP checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("report", type=Path)
    parser.add_argument("--managed", action="store_true")
    a = parser.parse_args()
    check(a.url, a.model, os.environ["THORUSTINFER_API_KEY"], a.report, a.managed)
