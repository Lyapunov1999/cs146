"""Simple CLI to call local Ollama llama3 with custom system/user prompts."""

import json
import sys
import re
from typing import Any, Dict, List, Callable, Tuple

import requests
from llama3 import chat  # 调用本地 llama3 接口

# Edit these two defaults to change prompts without retyping each run.

# TODO
DEFAULT_SYSTEM_PROMPT = "You are a powerful password validator."

# unused
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


# Ground-truth test suite used to evaluate generated code
SPECIALS = set("!@#$%^&*()-_")
TEST_CASES: List[Tuple[str, bool]] = [
    ("Password1!", True),       # valid
    ("password1!", False),      # missing uppercase
    ("Password!", False),       # missing digit
    ("Password1", False),       # missing special
]


def extract_code_block(text: str) -> str:
    m = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"```\n([\s\S]*?)```", text)
    if m:
        return m[-1].strip()
    return text.strip()


def load_function_from_code(code_str: str) -> Callable[[str], bool]:
    namespace: dict = {}
    exec(code_str, namespace)  # noqa: S102 (executing controlled code from model for exercise)
    func = namespace.get("is_valid_password")
    if not callable(func):
        raise ValueError("No callable is_valid_password found in generated code")
    return func


def evaluate_function(func: Callable[[str], bool]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for pw, expected in TEST_CASES:
        try:
            result = bool(func(pw))
        except Exception as exc:
            failures.append(f"Input: {pw} → raised exception: {exc}")
            continue

        if result != expected:
            # Compute diagnostic based on ground-truth rules
            reasons = []
            if len(pw) < 8:
                reasons.append("length < 8")
            if not any(c.islower() for c in pw):
                reasons.append("missing lowercase")
            if not any(c.isupper() for c in pw):
                reasons.append("missing uppercase")
            if not any(c.isdigit() for c in pw):
                reasons.append("missing digit")
            if not any(c in SPECIALS for c in pw):
                reasons.append("missing special")
            if any(c.isspace() for c in pw):
                reasons.append("has whitespace")

            failures.append(
                f"Input: {pw} → expected {expected}, got {result}. Failing checks: {', '.join(reasons) or 'unknown'}"
            )

    return (len(failures) == 0, failures)


def generate_initial_function(system_prompt: str) -> str:
    response = chat(
        model="llama3",
        system_prompt=system_prompt,
        user_prompt="Provide the implementation now.",
        temperature=0.2,
    )
    return extract_code_block(response)


# TODO: 在此填写用于 Reflexion 阶段的 system prompt
REFLEXION_PROMPT = ""


def your_build_reflexion_context(prev_code: str, failures: List[str]) -> str:
    # TODO: 如需调整反思上下文格式，可在此修改
    failure_text = "\n".join(f"- {f}" for f in failures) if failures else "- No failures"
    return f"""Previous attempt:
    ```
    {prev_code}
    ```

    Test feedback:
    {failure_text}

    Please return only the corrected implementation."""


def apply_reflexion(reflexion_prompt: str, prev_code: str, failures: List[str]) -> str:
    reflection_context = your_build_reflexion_context(prev_code, failures)
    response = chat(
        model="llama3",
        system_prompt=reflexion_prompt,
        user_prompt=reflection_context,
        temperature=0.2,
    )
    return extract_code_block(response)


def run_reflexion_flow() -> bool:
    # 1) Generate initial function
    initial_code = generate_initial_function(DEFAULT_SYSTEM_PROMPT)
    print("Initial code:\n" + initial_code)
    func = load_function_from_code(initial_code)
    passed, failures = evaluate_function(func)
    if passed:
        print("SUCCESS (initial implementation passed all tests)")
        return True
    print(f"FAILURE (initial implementation failed some tests): {failures}")

    # 2) Single reflexion iteration
    improved_code = apply_reflexion(REFLEXION_PROMPT, initial_code, failures)
    print("\nImproved code:\n" + improved_code)
    improved_func = load_function_from_code(improved_code)
    passed2, failures2 = evaluate_function(improved_func)
    if passed2:
        print("SUCCESS")
        return True

    print("Tests still failing after reflexion:")
    for f in failures2:
        print("- " + f)
    return False


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
    run_reflexion_flow()