"""Simple CLI to call local Ollama llama3 with custom system/user prompts."""

import json
import sys
from typing import Any, Dict, List

import requests

# Edit these two defaults to change prompts without retyping each run.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "Say hello in one sentence."


def chat(model: str, system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
	"""Send a chat request to the local Ollama server and return the response text."""

	messages: List[Dict[str, Any]] = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]

	try:
		resp = requests.post(
			"http://localhost:11434/api/chat",
			json={
				"model": model,
				"messages": messages,
				"options": {"temperature": temperature},
			},
			timeout=120,
			stream=True,
		)
	except requests.RequestException as exc:  # covers connection errors and timeouts
		raise SystemExit(f"Failed to reach Ollama server: {exc}") from exc

	if resp.status_code != 200:
		raise SystemExit(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

	# Stream chunks and stitch together the final message content
	final_text: List[str] = []
	for line in resp.iter_lines():
		if not line:
			continue
		try:
			payload = json.loads(line)
		except json.JSONDecodeError:
			continue  # skip malformed chunks

		# Each chunk has a "message" with partial content
		message = payload.get("message", {})
		chunk_text = message.get("content", "")
		if chunk_text:
			final_text.append(chunk_text)

		# When done, Ollama sends {"done": true}
		if payload.get("done"):
			break

	return "".join(final_text)


def run_llama3(system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
	"""Helper to send prompts to llama3, print, and return the reply."""

	reply = chat(model="llama3", system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
	print("\nModel reply:\n" + reply)
	return reply


def main() -> None:
	if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
		print(
			"Usage: python llama3.py [--interactive]\n"
			"Default: uses the prompts set in DEFAULT_SYSTEM_PROMPT and DEFAULT_USER_PROMPT.\n"
			"Interactive: add --interactive to type prompts at runtime."
		)
		return

	interactive = len(sys.argv) > 1 and sys.argv[1] in {"-i", "--interactive"}

	if interactive:
		system_prompt = input("System prompt: ").strip()
		user_prompt = input("User prompt: ").strip()
	else:
		system_prompt = DEFAULT_SYSTEM_PROMPT.strip()
		user_prompt = DEFAULT_USER_PROMPT.strip()

	if not user_prompt:
		raise SystemExit("User prompt cannot be empty.")

	run_llama3(system_prompt, user_prompt)


if __name__ == "__main__":
	main()
