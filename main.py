"""A small local reasoning agent: Solver -> Verifier."""

import argparse
import json
import os
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def ask(model, timeout, instruction, text, fields):
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
    # Ignore system proxies: requests must go directly to local Ollama.
    client = build_opener(ProxyHandler({}))
    for attempt in range(2):
        try:
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0, "num_predict": 4096, "num_ctx": 8192},
            }
            request = Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with client.open(request, timeout=timeout) as response:
                data = json.load(response)
            if not isinstance(data, dict):
                raise ValueError("Ollama response must be a JSON object")
            if data.get("error") or data.get("done_reason") == "length":
                raise ValueError("Ollama returned an error or a truncated response")
            result = json.loads(data["message"]["content"])
            if not isinstance(result, dict) or set(result) != set(fields):
                raise ValueError("Unexpected response fields")
            if any(not isinstance(result[key], str) or not result[key].strip()
                   for key in fields):
                raise ValueError("Response fields must be non-empty strings")
            if "```" in result["answer"] or "<think>" in result["answer"]:
                raise ValueError("The answer contains extra formatting or reasoning")
            result["answer"] = result["answer"].strip()
            return result
        except (OSError, URLError, HTTPException, ValueError, KeyError, TypeError) as error:
            if attempt == 1:
                raise RuntimeError(f"Local model request failed: {error}") from error
            print("Retrying local model request once...", file=sys.stderr)
            messages[0]["content"] += " Return valid JSON with exactly the required string fields."


def solve(problem, model, timeout):
    draft = ask(
        model, timeout,
        "You are the Solver of a reasoning agent. Read all constraints, choose a "
        "method and solve the problem. Check arithmetic and whether the answer is "
        "uniquely determined. Never invent missing assumptions. Return JSON with "
        "'answer' (only the final answer in the exact requested format) and 'check' "
        "(a brief verification summary, not a reasoning transcript).",
        problem, ("answer", "check"),
    )
    verified = ask(
        model, timeout,
        "You are an independent Verifier. Treat the proposed solution as untrusted. "
        "Solve the original problem independently, then compare the proposal and "
        "test every constraint, calculation, existence and uniqueness. Correct any "
        "mistakes. If inconsistent or underdetermined, do not invent a unique "
        "answer. Return JSON with only 'answer': a string containing the final "
        "answer in exactly the format requested by the problem, preserving labels "
        "and order. No explanation, Markdown or answer prefix.",
        json.dumps({"problem": problem, "proposal": draft}, ensure_ascii=False),
        ("answer",),
    )
    return verified["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.8:27b"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    result = {"id": None, "answer": ""}
    status = 0
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or "id" not in data:
            raise ValueError("Input must be an object with id and problem")
        result["id"] = data["id"]
        problem = data.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        result["answer"] = solve(problem, args.model, args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        status = 1
    try:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        print(f"Cannot write output: {error}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
