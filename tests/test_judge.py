import json
from types import SimpleNamespace

from finetune.evaluate import summarise_judge
from finetune.judge_generations import judge_labels, parse_args, read_rows, wait_seconds


class FakeRateLimit(Exception):
    pass


class FakeGroq:
    """Scores every answer 4/3; raises a rate limit on the calls listed in `limited_calls`."""

    def __init__(self, limited_calls=()):
        self.calls = 0
        self.limited_calls = set(limited_calls)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.calls in self.limited_calls:
            raise FakeRateLimit("Rate limit reached ... Please try again in 1m2.5s.")
        content = json.dumps({"correctness": 4, "helpfulness": 3})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def write_results(tmp_path, n_rows=3, judged=()):
    rows = []
    for i in range(n_rows):
        row = {"question": f"q{i}", "reference": f"r{i}", "prediction": f"p{i}", "context": ""}
        if i in judged:
            row["judge"] = {"correctness": 2.0, "helpfulness": 2.0}
        rows.append(row)
    (tmp_path / "generations_lora.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "eval_lora.json").write_text(json.dumps({"label": "lora", "rouge_l": 0.2, "judge_n": 1}))


def args_for(tmp_path, **overrides):
    return parse_args(["--labels", "lora", "--results-dir", str(tmp_path)] + sum(
        ([f"--{k.replace('_', '-')}", str(v)] for k, v in overrides.items()), []))


def test_wait_seconds_reads_groq_retry_hint():
    assert wait_seconds("Please try again in 4m19.632s.") == 4 * 60 + 19.632 + 5
    assert wait_seconds("try again in 7.5s") == 12.5
    assert wait_seconds("no hint") == 65


def test_summarise_judge_ignores_unjudged_rows_and_reports_a_ci():
    s = summarise_judge([{"correctness": 4.0, "helpfulness": 2.0}, None, {"correctness": 2.0, "helpfulness": 2.0}], "m")
    assert s["judge_n"] == 2 and s["judge_correctness"] == 3.0
    lo, hi = s["judge_correctness_ci95"]
    assert 2.0 <= lo <= 3.0 <= hi <= 4.0


def test_only_unjudged_rows_are_scored_and_eval_is_updated(tmp_path):
    write_results(tmp_path, judged={0})
    client = FakeGroq()
    assert judge_labels(client, args_for(tmp_path), FakeRateLimit, sleep=lambda s: None) == 0
    assert client.calls == 2                                   # row 0 was already judged
    result = json.loads((tmp_path / "eval_lora.json").read_text())
    assert result["judge_n"] == 3 and result["rouge_l"] == 0.2
    assert result["judge_correctness"] == round((2 + 4 + 4) / 3, 3)


def test_rate_limit_waits_then_retries(tmp_path):
    write_results(tmp_path)
    waits = []
    assert judge_labels(FakeGroq(limited_calls={2}), args_for(tmp_path), FakeRateLimit, sleep=waits.append) == 0
    assert waits == [67.5]
    assert all(r.get("judge") for r in read_rows(tmp_path / "generations_lora.jsonl"))


def test_stops_and_keeps_progress_when_waits_exceed_the_budget(tmp_path):
    write_results(tmp_path)
    code = judge_labels(FakeGroq(limited_calls={2, 3, 4}), args_for(tmp_path, max_wait_min=1),
                        FakeRateLimit, sleep=lambda s: None)
    assert code == 2
    rows = read_rows(tmp_path / "generations_lora.jsonl")
    assert [bool(r.get("judge")) for r in rows] == [True, False, False]
    assert json.loads((tmp_path / "eval_lora.json").read_text())["judge_n"] == 1
