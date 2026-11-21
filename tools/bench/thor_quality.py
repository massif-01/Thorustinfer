"""Score the bundled fixed-answer set using an installed CLI's ordinary offline batch entry."""

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFKC", text).strip().rstrip(".。!?！").strip().casefold()
    return re.sub(r"\s*,\s*", ",", " ".join(text.split()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("cli", "models", "dataset", "out", "gpu-lock"):
        ap.add_argument("--" + arg, required=True, type=Path)
    ap.add_argument("--profile", choices=("e8p", "w4a4"), required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(a.dataset.read_text())
    inputs = a.out / "input.jsonl"
    outputs = a.out / "output.jsonl"
    with inputs.open("w") as f:
        for id, prompt, _ in cases:
            body = dict(
                messages=[dict(role="user", content=prompt)],
                temperature=0,
                seed=20261002,
                max_completion_tokens=48,
                enable_thinking=False,
            )
            print(
                json.dumps(
                    dict(custom_id=id, method="POST", url="/v1/chat/completions", body=body),
                    ensure_ascii=False,
                ),
                file=f,
            )
    cmd = [
        str(a.cli),
        "batch",
        str(a.models),
        "--profile",
        a.profile,
        "--input",
        str(inputs),
        "--output",
        str(outputs),
        "--gpu-lock",
        str(a.gpu_lock),
        "--model",
        "qwen35-4b",
        "--mtp-drafts",
        "0",
        "--prefix-cache-mib",
        "0",
        "--max-active-requests",
        "4",
    ]
    with (a.out / "batch.log").open("w") as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = {r["custom_id"]: r for r in map(json.loads, outputs.read_text().splitlines())}
    results = []
    for id, prompt, answers in cases:
        body = rows[id]["response"]["body"]
        choice = body["choices"][0]
        text = choice["message"]["content"]
        ok = choice["finish_reason"] == "stop" and normalize(text) in map(normalize, answers)
        results.append(
            dict(
                id=id,
                prompt=prompt,
                accepted=answers,
                text=text,
                finish=choice["finish_reason"],
                usage=body["usage"],
                correct=ok,
            )
        )
    score = dict(
        profile=a.profile,
        correct=sum(r["correct"] for r in results),
        total=len(cases),
        results=results,
    )
    (a.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n")
    print(f"{a.profile} {score['correct']}/{score['total']}")
    # The batch CLI reports an output-budget stop with exit 1. It is a scored
    # wrong answer when every input received a valid response. Execution errors
    # and missing rows are still failures of the evaluation itself.
    complete = len(rows) == len(cases) and all(
        rows[key]["response"]["status_code"] == 200 for key, _, _ in cases
    )
    budget_stop = any(r["finish"] == "length" for r in results)
    if not (ret.returncode == 1 and complete and budget_stop):
        ret.check_returncode()


if __name__ == "__main__":
    main()
