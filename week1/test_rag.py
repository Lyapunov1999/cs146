import os
import re
from typing import List

from llama3 import chat

# 新增：本地语料与任务定义
DATA_FILES: List[str] = [os.path.join(os.path.dirname(__file__), "data", "api_docs.txt")]

def load_corpus_from_files(paths: List[str]) -> List[str]:
	corpus: List[str] = []
	for p in paths:
		if os.path.exists(p):
			try:
				with open(p, "r", encoding="utf-8") as f:
					corpus.append(f.read())
			except Exception as exc:
				corpus.append(f"[load_error] {p}: {exc}")
		else:
			corpus.append(f"[missing_file] {p}")
	return corpus

CORPUS: List[str] = load_corpus_from_files(DATA_FILES)
QUESTION = (
	"Write a Python function `fetch_user_name(user_id: str, api_key: str) -> str` that calls the documented API "
	"to fetch a user by id and returns only the user's name as a string."
)
REQUIRED_SNIPPETS = [
	"def fetch_user_name(",
	"requests.get",
	"/users/",
	"X-API-Key",
	"return",
]

def make_user_prompt(question: str, context_docs: List[str]) -> str:
	if context_docs:
		context_block = "\n".join(f"- {d}" for d in context_docs)
	else:
		context_block = "(no context provided)"
	return (
		f"Context (use ONLY this information):\n{context_block}\n\n"
		f"Task: {question}\n\n"
		"Requirements:\n"
		"- Use the documented Base URL and endpoint.\n"
		"- Send the documented authentication header.\n"
		"- Raise for non-200 responses.\n"
		"- Return only the user's name string.\n\n"
		"Output: A single fenced Python code block with the function and necessary imports.\n"
	)

## TO DO
DEFAULT_SYSTEM_PROMPT = "You are a senior Python engineer. Follow the task strictly."

DEFAULT_NUM_RUNS = 5
DEFAULT_TEMPERATURE = 0.0


def pick_context(corpus: List[str]) -> List[str]:
	"""选择用于回答的上下文，此处简单返回全部文档。"""
	return [doc for doc in corpus if doc.strip()]


def extract_code(text: str) -> str:
	"""提取回复中的最后一个 Python 代码块，若无则返回全文。"""
	python_blocks = re.findall(r"```python\n([\s\S]*?)```", text, flags=re.IGNORECASE)
	if python_blocks:
		return python_blocks[-1].strip()
	any_blocks = re.findall(r"```\n([\s\S]*?)```", text)
	if any_blocks:
		return any_blocks[-1].strip()
	return text.strip()


def run_tests(
	system_prompt: str = DEFAULT_SYSTEM_PROMPT,
	num_runs: int = DEFAULT_NUM_RUNS,
	temperature: float = DEFAULT_TEMPERATURE,
) -> bool:
	context_docs = pick_context(CORPUS)
	user_prompt = make_user_prompt(QUESTION, context_docs)

	for idx in range(num_runs):
		print(f"[{idx + 1}/{num_runs}] querying model...")
		reply = chat(model="llama3", system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
		code = extract_code(reply)
		missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in code]
		if not missing:
			print("SUCCESS\n" + code)
			return True
		print("Missing snippets:")
		for s in missing:
			print(f"  - {s}")
		print("Generated code preview:\n" + code)
	print("FAILED: required snippets not satisfied.")
	return False


if __name__ == "__main__":
	run_tests()
