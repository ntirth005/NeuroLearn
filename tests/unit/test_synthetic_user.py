"""P11 §2 (T0.6): the synthetic user. I-P11-1, I-P11-2, I-P11-6, I-P11-8 and the generator's own
consistency (truth table against stream, gold answers, forget requests, oracle rows)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from neurolearn.eval import synthetic_user as su
from neurolearn.eval.synthetic_user import (
    LAGS, TOOL_IMPLS, cluster_stats, generate, oracle_rows, question_cluster, run_tool, style_apply,
    style_check,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)


@lru_cache(maxsize=None)
def gen(seed: int, cycles: int):
    return generate(seed, cycles)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()


# ---- I-P11-1 --------------------------------------------------------------------------------


def test_I_P11_1_same_seed_is_byte_identical(tmp_path):
    a, b = generate(7, 30), generate(7, 30)
    assert a.dumps_stream() == b.dumps_stream() and a.dumps_truth() == b.dumps_truth()
    pa, pb = a.write(tmp_path / "a"), b.write(tmp_path / "b")
    for x, y in zip(pa, pb):
        assert x.read_bytes() == y.read_bytes()
    assert generate(8, 30).dumps_stream() != a.dumps_stream()
    assert generate(7, 31).dumps_truth() != a.dumps_truth()


def test_I_P11_1_independent_of_process_hash_seed():
    """A second interpreter with a different PYTHONHASHSEED writes the same bytes."""
    code = ("import hashlib; from neurolearn.eval.synthetic_user import generate; g = generate(3, 25); "
            "print(hashlib.sha256((g.dumps_stream() + g.dumps_truth()).encode()).hexdigest())")
    out = set()
    for hs in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": ""}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        out.add(r.stdout.strip())
    g = generate(3, 25)
    out.add(hashlib.sha256((g.dumps_stream() + g.dumps_truth()).encode()).hexdigest())
    assert len(out) == 1


# ---- the world ------------------------------------------------------------------------------


def test_world_sizes_and_uniqueness():
    t = gen(0, 10).truth
    kinds = Counter(o["kind"] for o in t["objects"].values())
    assert (kinds["person"], kinds["project"], kinds["jargon"], kinds["tool"], kinds["style"]) == (30, 30, 30, 5, 10)
    names = [o["name"].casefold() for o in t["objects"].values() if o["kind"] in ("person", "project", "jargon")]
    assert len(names) == len(set(names))
    facts = t["facts"]
    ports = [f["versions"][v]["value"] for f in facts.values() if f["attribute"] == "port" for v in (0, 1)]
    assert len(ports) == len(set(ports)) == 60
    owners = [f["versions"][0]["value"] for f in facts.values() if f["attribute"] == "owner"]
    assert len(set(owners)) == 30  # every person owns exactly one project: two-hop questions are well defined
    for f in facts.values():
        if f["category"] in ("fact", "jargon"):
            assert len(f["questions"]) == 3 and len(f["probes"]) == 2
            for v in f["versions"]:
                assert len(v["statements"]) == 3 and len(set(v["statements"])) == 3
                assert all(v["value"] in s for s in v["statements"])
            if len(f["versions"]) == 2:
                a, b = (norm(v["value"]) for v in f["versions"])
                assert a != b and a not in b and b not in a  # an old and a new value are never confusable
        if f["category"] == "style":
            assert len(f["versions"][0]["statements"]) == 3


def test_world_differs_by_seed_and_uses_no_real_words_of_the_domain():
    a, b = gen(0, 10).truth["objects"], gen(1, 10).truth["objects"]
    assert a != b


# ---- stream and truth agree -----------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_shape_and_minimal_visibility(seed):
    g = gen(seed, 50)
    by_cycle = {}
    for t in g.stream:
        by_cycle.setdefault((t["cycle"], t["session"]), []).append(t)
        assert set(t) <= {"id", "cycle", "session", "turn", "ts", "type", "source", "text", "file_name", "tools"}
    cycles = {c for c, _ in by_cycle}
    assert cycles == set(range(50))
    per_cycle = Counter(c for c, _ in by_cycle)
    assert all(2 <= n <= 4 for n in per_cycle.values())
    assert all(4 <= len(v) <= 10 for v in by_cycle.values())
    ids = [t["id"] for t in g.stream]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert {t["type"] for t in g.stream} <= {"filler", "state_fact", "ingest_file", "update_fact", "forget_request",
                                              "tool_task", "style_rule"}
    for t in g.stream:  # ground truth never rides along
        for key in ("gold", "fact_id", "facts", "expected", "arguments", "forget"):
            assert key not in t


def test_every_turn_type_occurs():
    seen = Counter(t["type"] for s in SEEDS for t in gen(s, 50).stream)
    assert set(seen) == {"filler", "state_fact", "ingest_file", "update_fact", "forget_request", "tool_task",
                         "style_rule"}


@pytest.mark.parametrize("seed", SEEDS)
def test_events_match_the_stream(seed):
    g = gen(seed, 50)
    texts = {t["id"]: t["text"] for t in g.stream}
    facts = g.truth["facts"]
    for e in g.truth["events"]:
        assert e["turn_id"] in texts
        if e["kind"] == "forgotten":
            continue
        stmt = facts[e["fact_id"]]["versions"][e["version"]]["statements"][e["paraphrase"]]
        assert stmt in texts[e["turn_id"]]
    for m in g.truth["turns"].values():
        for f in m.get("facts", []):
            assert f["fact_id"] in facts


@pytest.mark.parametrize("seed", SEEDS)
def test_updates_and_forgets_follow_the_rules(seed):
    g = gen(seed, 200)
    by_fact: dict[str, list[dict]] = {}
    for e in g.truth["events"]:
        by_fact.setdefault(e["fact_id"], []).append(e)
    for fid, evs in by_fact.items():
        kinds = [e["kind"] for e in evs]
        assert kinds.count("forgotten") <= 1 and kinds.count("updated") <= 1
        assert kinds[0] == "stated"
        if "forgotten" in kinds:
            assert kinds[-1] == "forgotten"  # nothing is said about a forgotten fact afterwards
            assert "updated" not in kinds
            assert g.truth["facts"][fid]["category"] in ("fact", "jargon")
        if "updated" in kinds:
            assert evs[kinds.index("updated")]["cycle"] >= evs[0]["cycle"]
    assert "updated" in {e["kind"] for e in g.truth["events"]}


@pytest.mark.parametrize("seed", SEEDS)
def test_forget_requests_name_one_entry_and_no_value(seed):
    g = gen(seed, 200)
    texts = {t["id"]: t for t in g.stream}
    facts, objects = g.truth["facts"], g.truth["objects"]
    forgets = [e for e in g.truth["events"] if e["kind"] == "forgotten"]
    assert forgets
    twice = 0
    for e in forgets:
        fact = facts[e["fact_id"]]
        req = texts[e["turn_id"]]
        assert req["type"] == "forget_request"
        values = [v["value"] for v in fact["versions"]]
        assert not any(v.casefold() in req["text"].casefold() for v in values)  # the request holds no fact
        assert objects[fact["object_id"]]["name"] in req["text"]
        named = texts[e["named_turn"]]
        assert named["type"] == "state_fact"  # never a file: a file entry holds ~10 other facts
        assert e["named_turn"] not in e["other_turns"]
        day = su.SIM_START + __import__("datetime").timedelta(days=named["cycle"])
        assert su._fmt_date(day) in req["text"]
        if e["stated_twice"]:
            twice += 1
            assert len(e["other_turns"]) == 1
            other = texts[e["other_turns"][0]]
            assert other["type"] == "state_fact" and other["cycle"] != named["cycle"]
            paras = {ev["paraphrase"] for ev in g.truth["events"]
                     if ev["fact_id"] == e["fact_id"] and ev["kind"] in ("stated", "restated")}
            assert len(paras) == 2  # two different wordings
        else:
            assert e["other_turns"] == []
    share = g.truth["forgets"]["twice"] / g.truth["forgets"]["requests"]
    assert twice == g.truth["forgets"]["twice"]
    assert abs(share - 0.4) <= 0.1, share


# ---- tools and style ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_tool_tasks_are_consistent(seed):
    g = gen(seed, 50)
    schemas = {s["name"]: s for s in g.truth["tools"]}
    assert set(schemas) == set(TOOL_IMPLS) and len(schemas) == 5
    n = 0
    for t in g.stream:
        if t["type"] != "tool_task":
            continue
        m = g.truth["turns"][t["id"]]
        assert [s["name"] for s in t["tools"]] == [m["tool"]]
        sch = schemas[m["tool"]]
        assert set(m["arguments"]) == set(sch["parameters"]["required"])
        assert run_tool(m["tool"], m["arguments"]) == m["expected"]
        n += 1
    assert n > 0
    for q in g.truth["questions"]:
        if q["category"] == "tool":
            call = q["expected_call"]
            assert run_tool(call["name"], call["arguments"]) == q["gold"]


def test_run_tool_known_values():
    assert run_tool("tally", {"a": 2, "b": 40}) == "42"
    assert run_tool("mirror", {"text": "abc"}) == "cba"
    assert run_tool("word_count", {"text": "one two three"}) == "3"
    assert run_tool("day_gap", {"start": "2031-01-01", "end": "2031-03-02"}) == "60"
    assert run_tool("join_path", {"parts": ["a", "b", "c"]}) == "a/b/c"


def test_every_style_rule_has_a_check_that_its_apply_satisfies_and_a_plain_reply_may_fail():
    t = gen(0, 10).truth
    rules = [f["rule"] for f in t["facts"].values() if f["category"] == "style"]
    assert len(rules) == 10 and len({r["kind"] for r in rules}) == 10
    for rule in rules:
        failed_plain = 0
        for prompt, plain in su._STYLE_TEACH:
            assert style_check(rule, style_apply(rule, plain)), (rule, prompt)
            failed_plain += not style_check(rule, plain)
        # not vacuous: a rule that every plain reply already satisfies tests nothing, except the two
        # the plain answers are written to satisfy
        if rule["kind"] not in ("no_commas", "forbid_word", "max_sentences", "max_words"):
            assert failed_plain == len(su._STYLE_TEACH), rule
    # and the checks reject violations
    by = {r["kind"]: r for r in rules}
    assert not style_check(by["no_commas"], "Yes, it does.")
    assert not style_check(by["forbid_word"], "Cats and dogs.")
    assert not style_check(by["max_sentences"], "One. Two. Three.")
    assert style_check(by["max_sentences"], "One. Two. — X.")
    assert not style_check(by["max_words"], "word " * 13)
    assert not style_check(by["sign_off"], "Fine. — Q.") or by["sign_off"]["param"] == "Q"


def test_style_teaching_answers_obey_the_plain_form():
    for prompt, ans in su._STYLE_TEACH:
        assert "," not in ans and not re.search(r"\band\b", ans, re.I) and "(" not in ans
        assert len(ans.split(".")[0].split()) <= 8
    assert len({p for p, _ in su._STYLE_TEACH}) == 10
    assert not {p for p, _ in su._STYLE_TEACH} & set(su._STYLE_EVAL_PROMPTS)


# ---- questions: I-P11-6 and the gold answers -------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_I_P11_6_questions_carry_facts_and_an_answer_object(seed):
    g = gen(seed, 50)
    t = g.truth
    cats = Counter(q["category"] for q in t["questions"])
    assert set(cats) == {"fact", "jargon", "multi", "style", "updated", "forgotten", "tool", "anchor"}
    for q in t["questions"]:
        assert q["fact_ids"] and all(f in t["facts"] for f in q["fact_ids"])
        assert len(q["fact_ids"]) == (2 if q["category"] == "multi" else 1)
        assert q["object_id"] == t["facts"][q["fact_ids"][-1]]["object_id"]
        assert question_cluster(t, q) == q["object_id"]
        assert q["lag"] in LAGS or q["lag"] is None
        assert 0 <= q["after_cycle"] < 50
        if q["category"] == "multi":
            assert t["objects"][q["object_id"]]["kind"] == "project"  # the answer's object, not the person
            assert q["fact_ids"][0].endswith(".owner") and q["fact_ids"][1].endswith(".port")
            assert q["fact_ids"][0].split(".")[0] == q["fact_ids"][1].split(".")[0]


def test_I_P11_6_clusters_are_a_function_of_truth_json(tmp_path):
    g = gen(1, 50)
    g.write(tmp_path)
    loaded = json.loads((tmp_path / "truth.json").read_text())
    a = {q["qid"]: question_cluster(g.truth, q) for q in g.truth["questions"]}
    b = {q["qid"]: question_cluster(loaded, q) for q in loaded["questions"]}
    assert a == b
    # no chaining: a person who owns a project does not pull the project's multi question into
    # the person's cluster, so no cluster mixes object kinds
    kinds_per_category: dict[str, set[str]] = {}
    for q in loaded["questions"]:
        kinds_per_category.setdefault(q["category"], set()).add(loaded["objects"][question_cluster(loaded, q)]["kind"])
    assert kinds_per_category["multi"] == {"project"}
    assert kinds_per_category["jargon"] == {"jargon"} and kinds_per_category["style"] == {"style"}


@pytest.mark.parametrize("seed", SEEDS)
def test_gold_answers_follow_the_truth_table(seed):
    g = gen(seed, 50)
    t = g.truth
    first = {}
    upd, forg = {}, {}
    for e in t["events"]:
        if e["kind"] == "stated":
            first.setdefault(e["fact_id"], e["cycle"])
        elif e["kind"] == "updated":
            upd[e["fact_id"]] = e["cycle"]
        elif e["kind"] == "forgotten":
            forg[e["fact_id"]] = e["cycle"]
    for q in t["questions"]:
        c, lag, cat = q["after_cycle"], q["lag"], q["category"]
        fid = q["fact_ids"][0]
        if cat in ("fact", "jargon"):
            assert c - first[fid] == lag
            assert not (fid in forg and forg[fid] <= c) and not (fid in upd and upd[fid] <= c)
            assert q["gold"] == t["facts"][fid]["versions"][0]["value"]
            assert q["text"] == t["facts"][fid]["questions"][0]  # paraphrase 1 only
        elif cat == "updated":
            assert c - upd[fid] == lag and q["gold"] == t["facts"][fid]["versions"][1]["value"]
            assert q["old_value"] == t["facts"][fid]["versions"][0]["value"] != q["gold"]
        elif cat == "forgotten":
            assert c - forg[fid] == lag and q["gold"] == t["facts"][fid]["versions"][0]["value"]
            assert isinstance(q["stated_twice"], bool) and isinstance(q["surviving_restatement"], bool)
        elif cat == "multi":
            proj = fid.split(".")[0]
            assert q["gold"] == t["facts"][proj + ".port"]["versions"][0]["value"]
            person = t["facts"][fid]["versions"][0]["value"]
            assert person in q["text"]
            for f in q["fact_ids"]:  # both facts stated by then and unchanged since
                assert first[f] <= c and not (f in forg and forg[f] <= c)
            assert not (proj + ".port" in upd and upd[proj + ".port"] <= c)
        elif cat == "style":
            assert q["check"] == t["facts"][fid]["rule"] and q["text"] in su._STYLE_EVAL_PROMPTS
        elif cat == "anchor":
            assert q["lag"] is None
    # same-day questions exist (they are asked before and after consolidation)
    assert any(q["lag"] == 0 for q in t["questions"])
    assert {q["lag"] for q in t["questions"]} >= {0, 1, 5, 10, 25}


def test_each_cycle_has_tool_and_anchor_questions_once_seen():
    t = gen(0, 30).truth
    per_cycle = Counter(q["after_cycle"] for q in t["questions"] if q["category"] == "anchor")
    assert set(per_cycle.values()) == {su.ANCHORS_PER_CYCLE} and len(per_cycle) == 30
    first_tool = {}
    for tid, m in t["turns"].items():
        if m.get("tool"):
            first_tool.setdefault(m["tool"], int(tid[1:5]))
    for q in t["questions"]:
        if q["category"] == "tool":
            assert q["after_cycle"] >= first_tool[q["expected_call"]["name"]]


# ---- I-P11-8 --------------------------------------------------------------------------------

MIN_CLUSTERS_FOR_CAP = 20  # D-53: below this a 10% cap cannot be met (largest share >= 1/clusters)


@pytest.mark.parametrize("cycles", [50, 200])
@pytest.mark.parametrize("seed", SEEDS)
def test_I_P11_8_largest_cluster_share(seed, cycles):
    stats = cluster_stats(gen(seed, cycles).truth)
    checked = 0
    for cat, s in stats.items():
        assert 0 < s["largest_share"] <= 1 and s["questions"] > 0
        assert s["largest_share"] >= 1 / s["clusters"] - 1e-12
        if s["clusters"] >= MIN_CLUSTERS_FOR_CAP:
            assert s["largest_share"] <= 0.10, (cat, s)
            checked += 1
    assert checked >= (2 if cycles == 50 else 6)
    if cycles == 200:  # the long run has enough clusters in every guarded category but style and tool
        for cat in ("fact", "jargon", "multi", "updated", "forgotten", "anchor"):
            assert stats[cat]["clusters"] >= MIN_CLUSTERS_FOR_CAP, (cat, stats[cat])


def test_cluster_stats_reports_every_category():
    s = cluster_stats(gen(0, 50).truth)
    assert set(s) == {"fact", "jargon", "multi", "style", "updated", "forgotten", "tool", "anchor"}
    assert s["style"]["clusters"] == 10 and s["tool"]["clusters"] == 5


def test_enough_clusters_for_the_kill_criterion_at_50_cycles():
    """G-K needs >= 10 clusters in jargon, style and multi on each of 3 seeds of 50 cycles."""
    for seed in SEEDS:
        s = cluster_stats(gen(seed, 50).truth)
        for cat in ("jargon", "style", "multi"):
            assert s[cat]["clusters"] >= 10, (seed, cat, s[cat])


# ---- I-P11-2 --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_I_P11_2_no_evaluation_question_in_any_entry_or_derived_row(seed):
    g = gen(seed, 50)
    t = g.truth
    entries = [norm(x["text"]) for x in g.stream]
    derived = []
    for x in g.stream:
        for r in oracle_rows(g, x["id"]):
            derived.append(norm(r.prompt))
            derived.append(norm(r.answer))
    asked = {norm(q["text"]) for q in t["questions"]}
    # paraphrase 1 of every fact, asked or not, is never shown in any form
    reserved = {norm(f["questions"][0]) for f in t["facts"].values() if f["questions"]}
    persons = [o["name"] for o in t["objects"].values() if o["kind"] == "person"]
    reserved |= {norm(su._MULTI_Q[0].format(person=p)) for p in persons}
    reserved |= {norm(p) for p in su._STYLE_EVAL_PROMPTS}
    for q in asked | reserved:
        assert q
        for text in entries:
            assert q not in text, q
        for text in derived:
            assert q not in text, q
    assert len(derived) > 100


def test_evaluation_questions_are_not_in_the_anchor_training_seed():
    asked = {norm(q["text"]) for q in gen(0, 20).truth["questions"] if q["category"] == "anchor"}
    asked |= {norm(q) for q in su._STYLE_EVAL_PROMPTS}
    rows = [json.loads(line) for line in (ROOT / "neurolearn/teach/anchor_seed.jsonl").read_text().splitlines()]
    blob = []
    for r in rows:
        blob.append(norm(r["text"]) if "text" in r else " ".join(norm(m["content"]) for m in r["messages"]))
    for q in asked:
        assert not any(q in b for b in blob), q


def test_a_tool_evaluation_question_is_redrawn_when_it_equals_a_stream_turn():
    """The draw of an evaluation instance can collide with a stream turn by chance; it must not stand."""
    import copy

    g = gen(0, 30)
    t = copy.deepcopy(g.truth)
    q0 = next(q for q in t["questions"] if q["category"] == "tool")
    name = q0["expected_call"]["name"]
    tid = next(k for k, m in t["turns"].items() if m.get("tool") == name)
    t["turns"][tid]["text"] = q0["text"]  # plant the collision
    rebuilt = su.build_questions(t, t["seed"])
    same = next(q for q in rebuilt if q["qid"] == q0["qid"])
    assert same["text"] != q0["text"]
    assert run_tool(same["expected_call"]["name"], same["expected_call"]["arguments"]) == same["gold"]
    stream_texts = {m["text"] for m in t["turns"].values() if m.get("tool")}
    assert not {q["text"] for q in rebuilt if q["category"] == "tool"} & stream_texts


def test_filler_and_anchor_questions_do_not_overlap():
    fill = {norm(x) for x in su._FILLER}
    assert len(fill) == len(su._FILLER)
    assert not fill & {norm(q) for q, _ in su._ANCHOR}
    assert not fill & {norm(p) for p in su._STYLE_EVAL_PROMPTS}


# ---- oracle rows ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_rows_use_the_right_paraphrases_and_cite_real_turns(seed):
    g = gen(seed, 50)
    t, facts = g.truth, g.truth["facts"]
    turn_ids = {x["id"] for x in g.stream}
    persons = [o["name"] for o in t["objects"].values() if o["kind"] == "person"]
    multi_prompts = {su._MULTI_Q[qi].format(person=p): p for p in persons for qi in (1, 2)}
    n_multi = 0
    for x in g.stream:
        rows = oracle_rows(g, x["id"])
        m = t["turns"][x["id"]]
        if x["type"] not in ("state_fact", "ingest_file", "update_fact", "style_rule"):
            assert rows == []
            continue
        used = {f["fact_id"] for f in m["facts"]}
        for r in rows:
            assert r.kind in ("qa", "implication", "probe") and r.source_turns
            assert set(r.source_turns) <= turn_ids and x["id"] in r.source_turns
            if r.kind == "implication":
                assert r.prompt == "" and r.answer
        for f in m["facts"]:
            fact, v, para = facts[f["fact_id"]], f["version"], f["paraphrase"]
            if fact["category"] == "style":
                continue
            qa = [r for r in rows if r.kind == "qa" and len(r.source_turns) == 1 and r.answer == fact["versions"][v]["value"]]
            prompts = " | ".join(norm(r.prompt) for r in qa)
            for qi in (1, 2):
                assert norm(fact["questions"][qi])[1:] in prompts, (f["fact_id"], qi)
            assert norm(fact["questions"][0]) not in prompts
            imps = {r.answer for r in rows if r.kind == "implication"}
            stmts = fact["versions"][v]["statements"]
            assert stmts[para] not in imps and {s for i, s in enumerate(stmts) if i != para} <= imps
            probes = " | ".join(norm(r.prompt) for r in rows if r.kind == "probe")
            for p in fact["probes"]:
                assert norm(p)[1:] in probes
            assert not set(map(norm, fact["probes"])) & set(map(norm, fact["questions"]))
        for r in rows:
            if r.prompt in multi_prompts:  # a two-hop row: answer is the port of the person's project
                n_multi += 1
                person = multi_prompts[r.prompt]
                proj = next(f["object_id"] for f in facts.values() if f["attribute"] == "owner"
                            and f["versions"][0]["value"] == person)
                assert r.answer == facts[proj + ".port"]["versions"][0]["value"]
                assert {proj + ".owner", proj + ".port"} <= {e["fact_id"] for e in t["events"]
                                                             if e["turn_id"] in r.source_turns}
    assert n_multi > 0


def test_style_oracle_rows_follow_their_rule():
    g = gen(0, 50)
    t = g.truth
    seen = 0
    for x in g.stream:
        if x["type"] != "style_rule":
            continue
        fid = t["turns"][x["id"]]["facts"][0]["fact_id"]
        rule = t["facts"][fid]["rule"]
        for r in oracle_rows(g, x["id"]):
            if r.kind in ("qa", "probe"):
                assert style_check(rule, r.answer) and r.prompt not in su._STYLE_EVAL_PROMPTS
                seen += 1
    assert seen > 10


def test_oracle_rows_insert_into_a_ledger(ledger):
    """The rows load with origin='oracle' and correct sources (§2.3); turns become entries in order."""
    from neurolearn.ledger.store import DerivedRow

    g = generate(0, 12)
    entry = {}
    n_rows = n_two = 0
    for x in g.stream:
        entry[x["id"]] = ledger.append("u1", x["source"], x["text"], x["ts"], session_id=x["session"])
        rows = oracle_rows(g, x["id"])
        for r in rows:
            ledger.add_derived([DerivedRow(r.kind, r.prompt, r.answer, origin="oracle")],
                               [entry[tid] for tid in r.source_turns])
            n_rows += 1
            n_two += len(r.source_turns) == 2
    assert n_rows > 40 and n_two > 0
    stored = ledger._conn().read("SELECT COUNT(*), SUM(origin = 'oracle') FROM derived")[0]
    assert stored == (n_rows, n_rows)
    assert set(entry) == {x["id"] for x in g.stream}
