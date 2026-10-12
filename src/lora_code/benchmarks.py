"""HumanEval and MBPP: prompts for an instruction-tuned model, and assembly
of the executable test program from a model response.

Both models in a comparison (base and fine-tuned) get identical prompts and
identical post-processing, so differences in pass@1 come from the weights.
"""

from __future__ import annotations

import re

FENCE_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)


def extract_code(response: str) -> str:
    """First fenced code block (an unterminated final fence is accepted, which
    happens when generation hits max_new_tokens); else the raw response."""
    m = FENCE_RE.search(response)
    return (m.group(1) if m else response).strip("\n")


# ------------------------------------------------------------------ HumanEval

def humaneval_instruction(problem: dict) -> str:
    return (
        "Complete the following Python function. Return the complete function, "
        "including the signature and any imports it needs, in a single ```python code block.\n\n"
        f"```python\n{problem['prompt'].rstrip()}\n```"
    )


def humaneval_program(problem: dict, response: str) -> str:
    code = extract_code(response)
    entry = problem["entry_point"]
    def_re = re.compile(rf"^def\s+{re.escape(entry)}\s*\(", re.MULTILINE)
    if def_re.search(code):
        # Keep the prompt's imports and helper functions (some problems define
        # helpers above the target), then the model's full function.
        m = def_re.search(problem["prompt"])
        prefix = problem["prompt"][: m.start()] if m else ""
        body = prefix + "\n" + code
    else:
        # The model returned only a body; graft it onto the given signature.
        body = problem["prompt"] + code
    return f"{body}\n\n{problem['test']}\n\ncheck({entry})\n"


# ----------------------------------------------------------------------- MBPP

def mbpp_instruction(problem: dict) -> str:
    # One test is shown so the required function name and signature are known
    # (the convention from Austin et al., 2021); the program runs all tests.
    return (
        f"{problem['prompt'].strip()}\n"
        f"Your code should pass this test:\n{problem['test_list'][0]}\n"
        "Return the complete solution in a single ```python code block."
    )


def mbpp_program(problem: dict, response: str) -> str:
    code = extract_code(response)
    imports = "\n".join(problem.get("test_imports") or [])
    tests = "\n".join(problem["test_list"])
    return f"{imports}\n{code}\n\n{tests}\n"


BENCHMARKS = {
    "humaneval": {
        "hf_path": "openai/openai_humaneval", "hf_name": None, "split": "test",
        "id_key": "task_id", "instruction": humaneval_instruction, "program": humaneval_program,
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp", "hf_name": "sanitized", "split": "test",
        "id_key": "task_id", "instruction": mbpp_instruction, "program": mbpp_program,
    },
}


def load_benchmark(name: str) -> list[dict]:
    from datasets import load_dataset
    spec = BENCHMARKS[name]
    ds = load_dataset(spec["hf_path"], spec["hf_name"], split=spec["split"])
    return [dict(r) for r in ds]


def benchmark_texts(problems: list[dict]) -> list[str]:
    """Text used to build the decontamination n-gram index."""
    out = []
    for p in problems:
        out.append(p.get("prompt", ""))
        out.append(p.get("canonical_solution", "") or p.get("code", ""))
    return out
