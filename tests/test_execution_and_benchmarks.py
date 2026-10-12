import time

from lora_code.benchmarks import extract_code, humaneval_program, mbpp_program
from lora_code.execution import FAILED, PASSED, TIMEOUT, run_many, run_program

HE = {
    "task_id": "HumanEval/X",
    "entry_point": "add",
    "prompt": 'from typing import List\n\n\ndef _helper(x):\n    return x\n\n\ndef add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
}
MBPP = {
    "task_id": 1,
    "prompt": "Write a function to find the maximum of a list.",
    "test_list": ["assert max_of([1, 3, 2]) == 3", "assert max_of([-5]) == -5"],
    "test_imports": [],
}


def test_run_program_statuses():
    assert run_program("assert 1 + 1 == 2").status == PASSED
    assert run_program("assert 1 + 1 == 3").status == FAILED
    assert run_program("def f(:\n  pass").status == FAILED
    t0 = time.time()
    assert run_program("while True:\n    pass", timeout=2).status == TIMEOUT
    assert time.time() - t0 < 6


def test_run_program_is_isolated_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_program("open('leak.txt', 'w').write('x')")
    assert not (tmp_path / "leak.txt").exists()


def test_memory_blowup_fails_cleanly():
    assert run_program("x = bytearray(8 * 1024**3)", timeout=5).status in (FAILED, TIMEOUT)


def test_run_many_preserves_order():
    res = run_many(["assert True", "assert False", "assert True"], workers=3)
    assert [r.status for r in res] == [PASSED, FAILED, PASSED]


def test_extract_code_variants():
    assert extract_code("Here:\n```python\nx = 1\n```\nbye") == "x = 1"
    assert extract_code("```py\ny = 2\n```") == "y = 2"
    assert extract_code("```\nz = 3\n```") == "z = 3"
    assert extract_code("```python\nunterminated = 1\n") == "unterminated = 1"
    assert extract_code("plain = 4") == "plain = 4"
    assert extract_code("```python\na = 1\n```\n```python\nb = 2\n```") == "a = 1"


def test_humaneval_full_function_response_passes():
    resp = "Sure.\n```python\ndef add(a: int, b: int) -> int:\n    return _helper(a) + b\n```"
    prog = humaneval_program(HE, resp)
    assert run_program(prog).status == PASSED  # helper from the prompt is kept


def test_humaneval_body_only_response_passes():
    assert run_program(humaneval_program(HE, "```python\n    return a + b\n```")).status == PASSED


def test_humaneval_wrong_answer_fails():
    resp = "```python\ndef add(a, b):\n    return a - b\n```"
    assert run_program(humaneval_program(HE, resp)).status == FAILED


def test_mbpp_programs():
    good = "```python\ndef max_of(xs):\n    return max(xs)\n```"
    bad = "```python\ndef max_of(xs):\n    return xs[0]\n```"
    assert run_program(mbpp_program(MBPP, good)).status == PASSED
    assert run_program(mbpp_program(MBPP, bad)).status == FAILED  # hidden test catches it
