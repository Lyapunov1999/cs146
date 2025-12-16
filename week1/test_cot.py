from llama3 import run_llama3
import re

#要求测五次有一次对就行

# Fill this!.
system_prompt = '''
You are a super talent mathematician who is excellent at modular arithmetic.
You are very careful and always show your work step by step. You are will decompose
large exponentiation problems into smaller parts using properties of modular arithmetic.
And remeber 3^{20} ≡ 1 (mod 100).
'''

system_prompt = '''
You are an expert mathematician specializing in modular arithmetic with a deep understanding of exponentiation techniques and modular properties. I seek your methodical approach to solving complex modular exponentiation problems with detailed step-by-step explanations.

Please address the problem by:

- Decomposing large exponents into smaller, manageable components using modular arithmetic theorems and properties.
- Applying relevant modular identities and theorems, such as Euler's theorem or Carmichael function, to simplify calculations.
- Using the given congruence 3^{20} ≡ 1 (mod 100) as a fundamental building block to reduce powers of 3 modulo 100.
- Demonstrating each step clearly, showing intermediate calculations and justifications for each transformation.
- Ensuring all reasoning is rigorous and transparent, suitable for advanced mathematical scrutiny.

Leverage your advanced expertise to provide a comprehensive solution that not only solves the problem efficiently but also serves as an instructive example of modular exponentiation techniques.
'''

############################################################################
# 这些也行
system_prompt = '''
You are will decompose large exponentiation problems into smaller parts using properties of modular arithmetic.
And remeber 3^{20} ≡ 1 (mod 100).
'''

system_prompt = '''
You are will decompose large exponentiation problems into smaller parts using properties of modular arithmetic.
You should keep in mind the important fact that 3^{20} ≡ 1 (mod 100).
'''

# 这两个就有时候不行了

# system_prompt = '''
# You are will decompose large exponentiation problems into smaller parts using properties of modular arithmetic.
# You must use the important fact that 3^{20} ≡ 1 (mod 100) in your calculations.
# '''

# system_prompt = '''
# You are will decompose large exponentiation problems into smaller parts using properties of modular arithmetic.
# '''


############################################################################


user_prompt = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

what is 3^{12345} (mod 100)?
"""

NUM_RUNS_TIMES = 5
EXPECTED_OUTPUT = "Answer: 43"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace."""
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def test_prompt():
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        output_text = run_llama3(system_prompt, user_prompt)
        final_answer = extract_final_answer(output_text)
        if final_answer.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {final_answer}")
    return False


if __name__ == "__main__":
    test_prompt()