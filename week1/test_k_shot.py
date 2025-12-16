from llama3 import chat

NUM_RUNS_TIMES = 5

# 这个llama模型估计搞不定

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = '''
You are an expert at reversing words. 
You should carefully reverse the letters in any given word one by one. 
Don't miss or add any letters.
Before you output the reversed word, make sure to double-check your work.
If the length doesn't match the input word, try again and think harder.

Here are some exaples.
<examples>
Input: appletown
Output: nwotelppa
</examples>
<examples>
Input: worldwide
Output: ediwdlrow
</examples>
<examples>
Input: smartphone
Output: enohptrams
</examples>
<examples>
Input: doublecheck
Output: kcehcelbuod
</examples>
'''

system_prompt='''
You are a strict string processor that only reverses the letters in a given word.

Keep in mind the following instructions when reversing words:
1. Carefully split the word into individual letters.
2. Reverse the order of the letters one by one.
3. Ensure that no letters are missed or added during the reversal process.
4. Before outputting the reversed word, double-check that the length matches the input word.

Don't do:
- Don't add or remove any letters.
- Don't output the answer if any unexpected errors occur.

Here are some examples:
<examples>
Input: appletown
Output: nwotelppa
</examples>
<examples>
Input: worldwide
Output: ediwdlrow
</examples>
<examples>
Input: smartphone
Output: enohptrams
</examples>
<examples>
Input: doublecheck
Output: kcehcelbuod
</examples>
'''

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""

EXPECTED_OUTPUT = "sutatsptth"


def test_with_llama3(system_prompt: str, temperature: float = 0.7) -> bool:
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        output_text = chat(
            model="llama3",
            system_prompt=system_prompt,
            user_prompt=USER_PROMPT,
            temperature=temperature,
        ).strip()
        if output_text == EXPECTED_OUTPUT:
            print("SUCCESS")
            return True
        print(f"Expected output: {EXPECTED_OUTPUT}")
        print(f"Actual output: {output_text}")
    return False


if __name__ == "__main__":
    test_with_llama3(YOUR_SYSTEM_PROMPT)
