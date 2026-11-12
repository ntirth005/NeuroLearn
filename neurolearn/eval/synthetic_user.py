"""Seeded synthetic user (P11 §2): a fictitious world, a multi-day stream of sessions, a
ground-truth table and the evaluation questions with their gold answers.

Everything is a pure function of ``(seed, cycles, StreamParams)``. Nothing here reads a clock, the
network or a model, and no model grades another model: gold answers are written by the generator.

Outputs (``Generated``):

* ``stream``: one dict per user turn, in order. This is *all the system under test may see*:
  ``id, cycle, session, turn, ts, type, source, text`` (+ ``file_name`` for files, ``tools`` for tool
  tasks). Ground truth is never in it.
* ``truth``: the world, every state change (stated / restated / updated / forgotten), the per-turn
  metadata, and every evaluation question (§3) with the ``fact_id``s it depends on, the
  ``object_id`` its answer belongs to (its bootstrap cluster, §5.1) and its gold answer.

``write`` stores them as ``stream.jsonl`` and ``truth.json``; the same seed gives byte-identical
files (I-P11-1).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
SIM_START = date(2031, 1, 6)  # simulated day 0; fictitious, so the base model cannot know it
LAGS = (0, 1, 5, 10, 25, 50)  # §3: same-day, then facts stated 1, 5, 10, 25, 50 cycles ago
EVAL_PARAPHRASE = 0  # question paraphrase 1 (index 0) is reserved for evaluation (§2.1)
ORACLE_PARAPHRASES = (1, 2)  # paraphrases 2-3 feed the oracle teacher (§2.3)
ANCHORS_PER_CYCLE = 10

N_PEOPLE, N_PROJECTS, N_JARGON, N_TOOLS, N_STYLE = 30, 30, 30, 5, 10  # P11 §2.1


@dataclass(frozen=True)
class StreamParams:
    """Shape of the stream. Properties of the benchmark, written into ``truth.json``.

    ``type_weights`` apply to each turn; a type that has nothing to act on yet (no live fact to
    update, say) becomes ``filler``. ``twice_forget_share`` is the share of forget requests that
    target a fact stated twice in two entries; when no such fact exists yet the turn restates a
    fact instead, so the supply builds up (D-53).
    """

    sessions: tuple[int, int] = (2, 4)
    turns: tuple[int, int] = (4, 10)
    type_weights: tuple[tuple[str, float], ...] = (
        ("filler", 0.875), ("state_fact", 0.026), ("update_fact", 0.014), ("forget_request", 0.012),
        ("tool_task", 0.04), ("style_rule", 0.025), ("ingest_file", 0.0015),
    )
    facts_per_statement: tuple[int, int] = (1, 3)
    file_facts: tuple[int, int] = (8, 12)
    restate_share: float = 0.15
    twice_forget_share: float = 0.4
    new_style_share: float = 0.85  # a style_rule turn states a new rule rather than repeating one
    priority: float = 0.35  # jargon facts and project (owner, port) pairs are introduced earlier


# --------------------------------------------------------------------------- vocabulary ---

_ONSET = ["b", "br", "d", "dr", "f", "g", "gl", "j", "k", "kr", "kv", "l", "m", "n", "p", "pl",
          "r", "s", "sk", "sl", "t", "tr", "v", "vr", "z", "zh", "zv"]
_VOWEL = ["a", "e", "i", "o", "u", "ai", "ou", "ia"]
_CODA = ["", "", "n", "x", "l", "r", "k", "m", "th", "sk", "nd"]

_LEVELS = ["junior", "senior", "staff", "principal", "lead"]
_FUNCTIONS = ["compiler engineer", "data steward", "release manager", "kernel developer",
              "security analyst", "build engineer", "support engineer", "platform architect"]
_PREF_ADJ = ["decaf", "smoked", "iced", "spiced", "salted", "herbal"]
_PREF_THING = ["mint tea", "oat milk", "dark chocolate", "ginger beer", "rye toast", "almond cake",
               "plum jam", "fig bars"]
_STATUSES = ["planning", "in review", "blocked", "shipping", "paused", "archived"]
_J_ADJ = ["nightly", "weekly", "manual", "staged", "emergency", "quarterly", "silent", "rolling"]
_J_NOUN = ["rebuild", "freeze", "handover", "backup", "audit", "rollback", "sync", "migration"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def _word(rng: random.Random) -> str:
    n = rng.choice((2, 2, 3))
    parts = [rng.choice(_ONSET) + rng.choice(_VOWEL) for _ in range(n - 1)]
    parts.append(rng.choice(_ONSET) + rng.choice(_VOWEL) + rng.choice(_CODA))
    return "".join(parts).capitalize()


class _Namer:
    """Invented words, unique across the whole world (case-insensitive)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.used = rng, set()

    def new(self) -> str:
        for _ in range(10_000):
            w = _word(self.rng)
            if 4 <= len(w) <= 9 and w.lower() not in self.used:
                self.used.add(w.lower())
                return w
        raise RuntimeError("could not invent a unique word")


def _fmt_date(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _rand_date(rng: random.Random) -> str:
    return _fmt_date(date(2032, 1, 1) + timedelta(days=rng.randrange(0, 700)))


# ------------------------------------------------------------------------------ templates ---
# (statements ×3, question paraphrases ×3, held-out probe paraphrases ×2). {o} = object name,
# {v} = value. Question paraphrase 0 is the evaluation question.
_T: dict[str, dict[str, Any]] = {
    "person.role": dict(
        s=["{o} works as {v}.", "These days {o} is the {v}.", "I should mention that {o}'s job title is {v}."],
        q=["What is {o}'s job title?", "What role does {o} have?", "Which position does {o} hold?"],
        p=["What does {o} do for a living here?", "{o} holds which job title?"]),
    "person.team": dict(
        s=["{o} is on the {v} team.", "{o} belongs to team {v}.", "You can find {o} in the {v} group."],
        q=["Which team is {o} on?", "What team does {o} belong to?", "{o} is a member of which team?"],
        p=["Name the team {o} works in.", "Where on the org chart does {o} sit?"]),
    "person.preference": dict(
        s=["{o} prefers {v}.", "If you are buying for {o}, get {v}.", "{o}'s favourite is {v}."],
        q=["What does {o} prefer?", "What should I get for {o}?", "What is {o}'s favourite?"],
        p=["Which treat does {o} like best?", "{o} would pick what?"]),
    "project.owner": dict(
        s=["{v} owns the {o} project.", "{o} is owned by {v}.", "The person responsible for {o} is {v}."],
        q=["Who owns the {o} project?", "Who is responsible for {o}?", "Whose project is {o}?"],
        p=["Name the owner of {o}.", "{o} belongs to which person?"]),
    "project.deadline": dict(
        s=["{o} is due on {v}.", "The deadline for {o} is {v}.", "We need to deliver {o} by {v}."],
        q=["When is {o} due?", "What is the deadline for {o}?", "By what date must {o} be delivered?"],
        p=["On which date does {o} have to ship?", "What is {o}'s due date?"]),
    "project.status": dict(
        s=["{o} is currently {v}.", "Status of {o}: {v}.", "Right now {o} is {v}."],
        q=["What is the status of {o}?", "What state is {o} in right now?", "Where does {o} stand at the moment?"],
        p=["Give me the current state of {o}.", "How would you describe {o}'s status?"]),
    "project.port": dict(
        s=["{o} listens on port {v}.", "The service for {o} runs on port {v}.", "Use port {v} to reach {o}."],
        q=["Which port does {o} use?", "What port does the {o} service listen on?", "On what port can I reach {o}?"],
        p=["Tell me the port number for {o}.", "{o} is served on which port?"]),
    "project.repo": dict(
        s=["The code for {o} lives in {v}.", "{o}'s repository is at {v}.", "You will find {o} under {v}."],
        q=["Where is the repository for {o}?", "Which path holds the {o} code?", "Where does {o}'s source live?"],
        p=["Give me the repo path for {o}.", "{o} is checked out at what path?"]),
    "jargon.definition": dict(
        s=["In the {proj} project, '{o}' means {v}.", "We say '{o}' when we mean {v} on {proj}.",
           "'{o}' is our word for {v} (on {proj})."],
        q=["What does '{o}' mean?", "What is meant by '{o}' in our team?", "Define '{o}' for me."],
        p=["Explain the term '{o}'.", "'{o}' refers to what?"]),
}
_MULTI_Q = [
    "Which port does the project owned by {person} use?",
    "What port is used by the project that {person} owns?",
    "On which port does {person}'s project run?",
]
_ATTR_LABEL = {"role": "job title", "team": "team", "preference": "preference", "owner": "owner",
               "deadline": "deadline", "status": "status", "port": "port", "repo": "repository",
               "definition": "meaning", "rule": "style rule"}
_TIME_DEPENDENT = {"deadline", "status", "port", "role", "team", "repo", "preference"}
_UPDATABLE = {"role", "team", "preference", "deadline", "status", "port", "repo"}

# ---------------------------------------------------------------------------- style rules ---


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s*—\s*\w\.\s*$", "", text.strip())  # a trailing sign-off is not a sentence
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def style_check(rule: dict[str, Any], reply: str) -> bool:
    """Rule-specific string check (P11 §5 style adherence). Pure function of the rule and reply."""
    kind, p, r = rule["kind"], rule.get("param"), reply.strip()
    if kind == "sign_off":
        return r.endswith(f"— {p}.")
    if kind == "opener":
        return r.startswith(f"{p}:")
    if kind == "max_sentences":
        return len(_sentences(r)) <= int(p)
    if kind in ("keyword", "phrase", "nickname"):
        return re.search(r"\b" + re.escape(str(p).lower()) + r"\b", r.lower()) is not None
    if kind == "no_commas":
        return "," not in r
    if kind == "max_words":
        return len(r.split()) <= int(p)
    if kind == "parenthetical":
        return re.search(r"\([^()]+\)", r) is not None
    if kind == "forbid_word":
        return re.search(r"\b" + re.escape(str(p)) + r"\b", r, re.I) is None
    raise ValueError(f"unknown style rule kind {kind!r}")


def style_apply(rule: dict[str, Any], text: str) -> str:
    """Rewrite a plain reply so it follows ``rule`` (used for oracle answers)."""
    kind, p = rule["kind"], rule.get("param")
    if kind == "sign_off":
        return f"{text} — {p}."
    if kind == "opener":
        return f"{p}: {text}"
    if kind == "max_sentences":
        return " ".join(_sentences(text)[: int(p)])
    if kind == "keyword":
        return f"{text} Keep {p} in mind."
    if kind == "phrase":
        return f"{str(p).capitalize()}: {text}"
    if kind == "nickname":
        return f"{text} Over to you {p}."
    if kind == "no_commas":
        return text.replace(",", "")
    if kind == "max_words":
        first = _sentences(text)[0]
        return first if len(first.split()) <= int(p) else " ".join(first.split()[: int(p)]) + "."
    if kind == "parenthetical":
        return f"{text} (noted)"
    if kind == "forbid_word":
        return re.sub(r"\s*\b" + re.escape(str(p)) + r"\b", "", text, flags=re.I)
    raise ValueError(f"unknown style rule kind {kind!r}")


# Evaluation prompts (group A) and oracle-teacher prompts with canned plain answers (group B).
# Plain answers have no commas, no "and", no parentheses; the first sentence has at most 8 words.
_STYLE_EVAL_PROMPTS = [
    "Explain what a hash table is.", "Give me a quick summary of how vaccines work.",
    "What should I know about starting to run?", "Tell me about the Roman aqueducts.",
    "How does a microwave oven heat food?", "Why do cats purr?", "What is inflation?",
    "Describe how a bill becomes law.", "Why do onions make people cry?",
    "What is the greenhouse effect?",
]
_STYLE_TEACH = [
    ("How does a thermometer work?", "It measures heat. A liquid expands as it warms. The rise shows the temperature."),
    ("What is a tide?", "It is a daily sea rise. The moon pulls on the ocean. Land shapes the pattern."),
    ("Why do leaves change colour?", "Green fades in autumn. Trees stop making chlorophyll. Other pigments then show through."),
    ("What does a battery do?", "It stores energy. Chemical reactions release current. Devices then draw that power."),
    ("How do bees make honey?", "Bees gather nectar. They pass it mouth to mouth. Wings then dry it thick."),
    ("What is an echo?", "It is reflected sound. Sound waves bounce off hard walls. You hear them again later."),
    ("Why is ice slippery?", "A thin film forms. Pressure melts the top layer. Shoes glide across it."),
    ("How does soap clean?", "Soap grabs grease. One end loves water. The other end loves oil."),
    ("What causes thunder?", "Lightning heats the air. The air expands fast. We hear that as thunder."),
    ("What is a prime number?", "It has two divisors. One is itself. The other is one."),
]


def _style_rules(rng: random.Random, namer: _Namer) -> list[dict[str, Any]]:
    letter = rng.choice("ABCDEFGHJKLMPRSTVW")
    opener, keyword, nick = namer.new(), namer.new(), namer.new()
    phrase = f"by {namer.new().lower()} reckoning"
    rules = [
        ("sign_off", letter, [f"Always sign off your replies with '— {letter}.'",
                              f"End every answer you give me with '— {letter}.' please.",
                              f"From now on, finish each reply with the signature '— {letter}.'"]),
        ("opener", opener, [f"Start every reply with '{opener}:'.", f"Open each answer with the word {opener} and a colon.",
                            f"Please begin all of your replies with '{opener}:'."]),
        ("max_sentences", 2, ["Answer in at most two sentences.", "Keep every reply to two sentences or fewer.",
                              "No more than two sentences per answer, please."]),
        ("keyword", keyword, [f"Work the word '{keyword}' into every reply.", f"Every answer should contain the word {keyword}.",
                              f"Make sure '{keyword}' shows up in each of your replies."]),
        ("no_commas", None, ["Do not use commas in your replies.", "Write every answer without a single comma.",
                             "Never put a comma in anything you say to me."]),
        ("max_words", 12, ["Keep every answer under thirteen words.", "Answer in twelve words at most.",
                           "Replies must be twelve words or shorter."]),
        ("phrase", phrase, [f"Include the phrase '{phrase}' in every answer.", f"Each reply must say '{phrase}' somewhere.",
                            f"Please always slip '{phrase}' into what you write."]),
        ("nickname", nick, [f"Call me '{nick}' in every reply.", f"Address me as {nick} each time you answer.",
                            f"My nickname is {nick}, so use it in all your replies."]),
        ("parenthetical", None, ["Add a short remark in parentheses to every reply.", "Every answer needs a brief aside in brackets.",
                                 "End each reply with a short parenthetical note."]),
        ("forbid_word", "and", ["Never use the word 'and' in a reply.", "The word 'and' is banned in your answers.",
                                "Do not write the word and anywhere in a reply."]),
    ]
    assert len(rules) == N_STYLE
    return [{"kind": k, "param": p, "statements": s} for k, p, s in rules]


# ------------------------------------------------------------------------------- tools ---


def _tally(a: int, b: int) -> str:
    return str(a + b)


def _mirror(text: str) -> str:
    return text[::-1]


def _word_count(text: str) -> str:
    return str(len(text.split()))


def _day_gap(start: str, end: str) -> str:
    return str((date.fromisoformat(end) - date.fromisoformat(start)).days)


def _join_path(parts: list[str]) -> str:
    return "/".join(parts)


TOOL_IMPLS: dict[str, Callable[..., str]] = {
    "tally": _tally, "mirror": _mirror, "word_count": _word_count, "day_gap": _day_gap,
    "join_path": _join_path,
}


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Deterministic fake implementation of a world tool."""
    return TOOL_IMPLS[name](**arguments)


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS = [
    _schema("tally", "Add two whole numbers.", {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]),
    _schema("mirror", "Reverse a piece of text.", {"text": {"type": "string"}}, ["text"]),
    _schema("word_count", "Count the words in a piece of text.", {"text": {"type": "string"}}, ["text"]),
    _schema("day_gap", "Days between two ISO dates.",
            {"start": {"type": "string"}, "end": {"type": "string"}}, ["start", "end"]),
    _schema("join_path", "Join path segments with slashes.",
            {"parts": {"type": "array", "items": {"type": "string"}}}, ["parts"]),
]
_SENT_WORDS = ["river", "lamp", "quiet", "orange", "table", "slowly", "paper", "window", "garden", "silver",
               "morning", "bridge", "carry", "small", "engine", "cloud", "sing", "marble", "north", "ladder"]
_SEGMENTS = ["alpha", "build", "data", "logs", "tmp", "docs", "cache", "bin", "assets", "notes"]


def _tool_instance(tool: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    if tool == "tally":
        a, b = rng.randrange(11, 990), rng.randrange(11, 990)
        return f"What is {a} plus {b}?", {"a": a, "b": b}
    if tool == "mirror":
        w = " ".join(rng.sample(_SENT_WORDS, 3))
        return f"Please reverse the text '{w}'.", {"text": w}
    if tool == "word_count":
        s = " ".join(rng.sample(_SENT_WORDS, rng.randint(4, 9)))
        return f"How many words are in this sentence: '{s}'?", {"text": s}
    if tool == "day_gap":
        d1 = date(2031, 1, 1) + timedelta(days=rng.randrange(0, 300))
        d2 = d1 + timedelta(days=rng.randrange(2, 200))
        return f"How many days are there from {d1.isoformat()} to {d2.isoformat()}?", \
            {"start": d1.isoformat(), "end": d2.isoformat()}
    if tool == "join_path":
        parts = rng.sample(_SEGMENTS, 4)
        return f"Build a path from {parts[0]}, {parts[1]}, {parts[2]} and {parts[3]}.", {"parts": parts}
    raise ValueError(tool)


# ------------------------------------------------------------------- filler and anchors ---

_FILLER = [
    "Can you help me plan a weekend hike?", "I am trying to decide what to cook tonight.",
    "What is a good way to stay focused while reading?", "Tell me a fun fact about octopuses.",
    "I slept badly last night. Any tips?", "How do I politely decline a meeting invitation?",
    "Suggest a name for a small vegetable garden.", "What is a good stretching routine for desk work?",
    "I need ideas for a birthday present for a friend who likes maps.",
    "Can you explain how compound interest works?", "What should I pack for a rainy city trip?",
    "Give me a three-item checklist for a tidy workspace.", "Why do people enjoy puzzles so much?",
    "Help me think of a short toast for a colleague's farewell.", "How long should I steep green tea?",
    "I want to learn to juggle. Where do I start?", "What makes a paragraph easy to read?",
    "Recommend a calm playlist theme for a long train ride.", "Is it better to walk or cycle to work?",
    "How do I keep a houseplant alive through winter?", "What is a sensible way to split a restaurant bill?",
    "Write a two-line rhyme about coffee.", "How do I get better at remembering names?",
    "What are some low-effort ways to save energy at home?", "Describe a relaxing evening routine.",
    "Is it worth keeping a daily journal?", "I am bored. Give me a quick brain teaser.",
    "How can I make my morning commute less stressful?", "What is the best way to learn a new language slowly?",
    "Tell me something interesting about bridges.", "How do I stop procrastinating on small chores?",
    "Suggest a theme for a casual team lunch.", "What does a good weekly plan look like?",
    "How can I explain my job to a ten-year-old?", "Which season is best for a first camping trip?",
    "Give me a pep talk before a big presentation.", "Why do clocks run clockwise?",
    "What are good questions to ask at a job interview?", "I want to start a book club. Advice?",
    "Name a few quiet hobbies for a rainy afternoon.",
]
_ANCHOR = [
    ("What is the capital of Germany?", "Berlin"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Canada?", "Ottawa"), ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Kenya?", "Nairobi"), ("What is the capital of Peru?", "Lima"),
    ("What is the capital of Norway?", "Oslo"), ("What is the capital of Greece?", "Athens"),
    ("What is the chemical symbol for gold?", "Au"), ("In what state is water at room temperature?", "liquid"),
    ("Which planet is known as the red planet?", "Mars"), ("Which planet is closest to the Sun?", "Mercury"),
    ("Which gas do plants absorb for photosynthesis?", "carbon dioxide"),
    ("Which force pulls objects towards the Earth?", "gravity"), ("What is at the centre of an atom?", "nucleus"),
    ("What is the hardest natural substance?", "diamond"), ("What is the closest star to Earth?", "Sun"),
    ("Which organ pumps blood around the body?", "heart"),
    ("Who wrote the play Hamlet?", "Shakespeare"), ("Who wrote Pride and Prejudice?", "Austen"),
    ("Who composed The Four Seasons?", "Vivaldi"), ("What is the opposite of tall?", "short"),
    ("What is the plural of mouse?", "mice"), ("What is the past tense of run?", "ran"),
    ("Which language is mainly spoken in Brazil?", "Portuguese"), ("What currency does Japan use?", "yen"),
    ("Which language did the ancient Romans speak?", "Latin"), ("Who wrote the novel 1984?", "Orwell"),
    ("What is the largest ocean on Earth?", "Pacific"), ("What is the longest river in Africa?", "Nile"),
    ("What is the tallest mountain on Earth?", "Everest"), ("On which continent is Kenya?", "Africa"),
    ("Which fruit is said to keep the doctor away?", "apple"), ("Which season comes after winter?", "spring"),
    ("What is the main ingredient of bread?", "flour"), ("Which metal is liquid at room temperature?", "mercury"),
    ("Who is credited with inventing the telephone?", "Bell"), ("Which sport is played at Wimbledon?", "tennis"),
]


# ------------------------------------------------------------------------------- world ---


def _fact(obj: str, attr: str, cat: str, versions: list[tuple[str, list[str]]], tpl: dict[str, Any] | None,
          name: str, extra: dict[str, Any] | None = None, **fmt: str) -> dict[str, Any]:
    f: dict[str, Any] = {
        "fact_id": f"{obj}.{attr}", "object_id": obj, "category": cat, "attribute": attr,
        "label": _ATTR_LABEL.get(attr, attr), "time_dependent": attr in _TIME_DEPENDENT,
        "updatable": len(versions) > 1,
        "questions": [t.format(o=name, **fmt) for t in tpl["q"]] if tpl else [],
        "probes": [t.format(o=name, **fmt) for t in tpl["p"]] if tpl else [],
        "versions": [{"value": v, "statements": s} for v, s in versions],
    }
    f.update(extra or {})
    return f


def _versions(tpl: dict[str, Any], name: str, values: list[str], **fmt: str) -> list[tuple[str, list[str]]]:
    return [(v, [t.format(o=name, v=v, **fmt) for t in tpl["s"]]) for v in values]


def build_world(seed: int) -> dict[str, Any]:
    """The fictitious world: objects, facts (with a replacement value for updatable ones), tools."""
    rng = random.Random(f"neurolearn-world:{seed}")
    namer = _Namer(rng)
    objects: dict[str, dict[str, str]] = {}
    facts: dict[str, dict[str, Any]] = {}

    def add(f: dict[str, Any]) -> None:
        facts[f["fact_id"]] = f

    def other(rng_: random.Random, pool: list[str], v0: str) -> str:
        return rng_.choice([x for x in pool if x != v0])

    teams = [namer.new() for _ in range(8)]
    roles = [f"{a} {b}" for a in _LEVELS for b in _FUNCTIONS]
    prefs = [f"{a} {b}" for a in _PREF_ADJ for b in _PREF_THING]
    people_names, firsts = [], set()
    while len(people_names) < N_PEOPLE:
        first, last = namer.new(), namer.new()
        people_names.append(f"{first} {last}")
        firsts.add(first)
    role_pick, pref_pick = rng.sample(roles, N_PEOPLE), rng.sample(prefs, N_PEOPLE)
    for i, name in enumerate(people_names):
        oid = f"person:{i:02d}"
        objects[oid] = {"kind": "person", "name": name}
        for attr, v0, pool in (("role", role_pick[i], roles), ("team", rng.choice(teams), teams),
                               ("preference", pref_pick[i], prefs)):
            tpl = _T[f"person.{attr}"]
            add(_fact(oid, attr, "fact", _versions(tpl, name, [v0, other(rng, pool, v0)]), tpl, name))

    proj_names = [namer.new() for _ in range(N_PROJECTS)]
    owner_perm = rng.sample(range(N_PEOPLE), N_PEOPLE)  # each person owns exactly one project (multi)
    ports = rng.sample(range(10_000, 65_000), 2 * N_PROJECTS)
    repos = rng.sample([f"/srv/{a.lower()}/{b.lower()}" for a, b in
                        ((namer.new(), namer.new()) for _ in range(2 * N_PROJECTS))], 2 * N_PROJECTS)
    for i, name in enumerate(proj_names):
        oid = f"project:{i:02d}"
        objects[oid] = {"kind": "project", "name": name}
        owner = people_names[owner_perm[i]]
        tpl = _T["project.owner"]
        add(_fact(oid, "owner", "fact", _versions(tpl, name, [owner]), tpl, name))
        d0 = _rand_date(rng)
        tpl = _T["project.deadline"]
        add(_fact(oid, "deadline", "fact", _versions(tpl, name, [d0, _rand_date(rng)]), tpl, name))
        s0 = rng.choice(_STATUSES)
        tpl = _T["project.status"]
        add(_fact(oid, "status", "fact", _versions(tpl, name, [s0, other(rng, _STATUSES, s0)]), tpl, name))
        tpl = _T["project.port"]
        add(_fact(oid, "port", "fact", _versions(tpl, name, [str(ports[i]), str(ports[N_PROJECTS + i])]), tpl, name))
        tpl = _T["project.repo"]
        add(_fact(oid, "repo", "fact", _versions(tpl, name, [repos[i], repos[N_PROJECTS + i]]), tpl, name))

    pairs = rng.sample([(a, b) for a in _J_ADJ for b in _J_NOUN], N_JARGON)
    for i in range(N_JARGON):
        oid = f"jargon:{i:02d}"
        term = namer.new().lower()
        objects[oid] = {"kind": "jargon", "name": term}
        proj = proj_names[rng.randrange(N_PROJECTS)]
        tpl = _T["jargon.definition"]
        add(_fact(oid, "definition", "jargon", _versions(tpl, term, [f"{pairs[i][0]} {pairs[i][1]}"], proj=proj),
                  tpl, term, extra={"project": proj}, proj=proj))

    for i, rule in enumerate(_style_rules(rng, namer)):
        oid = f"style:{i:02d}"
        objects[oid] = {"kind": "style", "name": f"{rule['kind']} rule"}
        f = _fact(oid, "rule", "style", [(rule["kind"], rule["statements"])], None, objects[oid]["name"],
                  extra={"rule": {"kind": rule["kind"], "param": rule["param"]}})
        add(f)

    tools = []
    for i in range(N_TOOLS):
        oid = f"tool:{i}"
        sch = TOOL_SCHEMAS[i]
        objects[oid] = {"kind": "tool", "name": sch["name"]}
        tools.append(sch)
        add(_fact(oid, "use", "tool", [(sch["name"], [])], None, sch["name"]))

    for i, (q, a) in enumerate(_ANCHOR):
        oid = f"anchor:{i:02d}"
        objects[oid] = {"kind": "anchor", "name": f"anchor {i}"}
        f = _fact(oid, "answer", "anchor", [(a, [])], None, "anchor")
        f["questions"] = [q]
        add(f)

    return {"objects": objects, "facts": facts, "tools": tools, "people": people_names}


# ------------------------------------------------------------------------------ stream ---


@dataclass
class Generated:
    seed: int
    cycles: int
    stream: list[dict[str, Any]]
    truth: dict[str, Any]

    def dumps_stream(self) -> str:
        return "".join(json.dumps(t, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                       for t in self.stream)

    def dumps_truth(self) -> str:
        return json.dumps(self.truth, sort_keys=True, ensure_ascii=False, indent=1) + "\n"

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stream.jsonl").write_text(self.dumps_stream(), encoding="utf-8", newline="\n")
        (d / "truth.json").write_text(self.dumps_truth(), encoding="utf-8", newline="\n")
        return d / "stream.jsonl", d / "truth.json"


class _FactState:
    __slots__ = ("stated", "version", "forgotten", "via_file", "updated_cycle", "forgotten_cycle")

    def __init__(self) -> None:
        self.stated: list[tuple[str, int, int, int]] = []  # (turn_id, cycle, version, paraphrase)
        self.version = 0
        self.forgotten = False
        self.via_file = False
        self.updated_cycle: int | None = None
        self.forgotten_cycle: int | None = None


def _pick_weighted(rng: random.Random, weights: tuple[tuple[str, float], ...]) -> str:
    total = sum(w for _, w in weights)
    x = rng.random() * total
    for name, w in weights:
        x -= w
        if x < 0:
            return name
    return weights[-1][0]


def _lead(rng: random.Random) -> str:
    return rng.choice(["", "", "By the way, ", "Quick note: ", "Also, ", "For the record, ", "Just so you know, "])


def generate(seed: int, cycles: int, params: StreamParams | None = None) -> Generated:
    """Build the world, the stream and the ground-truth table for ``cycles`` simulated days."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    P = params or StreamParams()
    world = build_world(seed)
    facts, objects = world["facts"], world["objects"]
    rng = random.Random(f"neurolearn-stream:{seed}:{cycles}")
    state = {fid: _FactState() for fid, f in facts.items() if f["category"] in ("fact", "jargon")}
    style_state = {fid: _FactState() for fid, f in facts.items() if f["category"] == "style"}
    # Unstated facts, consumed from the end. The world has ~270 stated facts for 200 cycles, so
    # facts arrive at ~1.3 per cycle; jargon terms and each project's (owner, port) pair come
    # earlier on average so that `jargon` and `multi` have enough clusters in a 50-cycle run.
    groups: list[tuple[float, list[str]]] = []
    for fid in sorted(state):
        obj = facts[fid]["object_id"]
        if facts[fid]["attribute"] == "port":
            continue  # travels with the owner fact
        members = [fid, f"{obj}.port"] if facts[fid]["attribute"] == "owner" else [fid]
        pri = facts[fid]["attribute"] == "owner" or facts[fid]["category"] == "jargon"
        groups.append((rng.random() * (P.priority if pri else 1.0), members))
    groups.sort(key=lambda g: g[0])
    pool = [fid for _, members in reversed(groups) for fid in reversed(members)]
    style_pool = sorted(style_state)
    rng.shuffle(style_pool)
    tools_seen: dict[str, int] = {}  # tool name -> first cycle
    stream: list[dict[str, Any]] = []
    turns_meta: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    forget_stats = {"requests": 0, "twice": 0}
    tool_used: dict[tuple[str, int], int] = {}

    def statement(fid: str, version: int, para: int) -> str:
        return facts[fid]["versions"][version]["statements"][para]

    def ev(kind: str, cyc: int, tid: str, fid: str, version: int, para: int, **more: Any) -> None:
        events.append({"cycle": cyc, "turn_id": tid, "fact_id": fid, "kind": kind, "version": version,
                       "paraphrase": para, **more})

    for cyc in range(cycles):
        day = SIM_START + timedelta(days=cyc)
        n_sessions = rng.randint(*P.sessions)
        for sess in range(n_sessions):
            n_turns = rng.randint(*P.turns)
            for t in range(n_turns):
                tid = f"c{cyc:04d}.s{sess}.t{t:02d}"
                ts = datetime(day.year, day.month, day.day, 8 + 4 * sess, 5 * t, tzinfo=timezone.utc) \
                    .isoformat().replace("+00:00", "Z")
                kind = _pick_weighted(rng, P.type_weights)
                rec: dict[str, Any] = {"id": tid, "cycle": cyc, "session": f"c{cyc:04d}.s{sess}", "turn": t,
                                       "ts": ts, "source": "chat"}
                meta: dict[str, Any] = {}

                live = [f for f, s in state.items() if s.stated and not s.forgotten]
                chat_once = [f for f in live if len(state[f].stated) == 1 and not state[f].via_file
                             and state[f].updated_cycle is None]
                chat_twice = [f for f in live if len(state[f].stated) == 2 and not state[f].via_file
                              and state[f].updated_cycle is None
                              and state[f].stated[0][1] != state[f].stated[1][1]]
                restatable = [f for f in chat_once if state[f].stated[0][1] != cyc]

                def restate() -> bool:
                    if not restatable:
                        return False
                    fid = rng.choice(restatable)
                    first_para = state[fid].stated[0][3]
                    para = rng.choice([p for p in range(3) if p != first_para])
                    state[fid].stated.append((tid, cyc, state[fid].version, para))
                    rec.update(type="state_fact", text=_lead(rng) + statement(fid, state[fid].version, para))
                    meta["facts"] = [{"fact_id": fid, "version": state[fid].version, "paraphrase": para,
                                      "restated": True}]
                    ev("restated", cyc, tid, fid, state[fid].version, para)
                    return True

                if kind == "forget_request":
                    # a fixed share of requests target a fact stated twice: request k is of that
                    # kind when floor((k+1)·share) > floor(k·share); a request that cannot be
                    # served yet restates a fact instead and the same k is tried again next time
                    k = forget_stats["requests"]
                    want_twice = int((k + 1) * P.twice_forget_share) > int(k * P.twice_forget_share)
                    if want_twice and not chat_twice:
                        if restate():
                            kind = "_done"
                        else:
                            kind = "filler"
                    elif not want_twice and not chat_once:
                        kind = "filler"
                    elif want_twice:
                        fid = rng.choice(chat_twice)
                        stated = state[fid].stated
                        named = rng.choice(stated)
                        others = [s[0] for s in stated if s[0] != named[0]]
                        _do_forget(rec, meta, state, facts, objects, fid, named, others, True, cyc, tid, ev)
                        forget_stats["requests"] += 1
                        forget_stats["twice"] += 1
                        kind = "_done"
                    else:
                        fid = rng.choice(chat_once)
                        named = state[fid].stated[0]
                        _do_forget(rec, meta, state, facts, objects, fid, named, [], False, cyc, tid, ev)
                        forget_stats["requests"] += 1
                        kind = "_done"
                elif kind == "state_fact":
                    if restatable and rng.random() < P.restate_share:
                        restate()
                        kind = "_done"
                    elif not pool:
                        kind = "filler" if not restate() else "_done"
                    else:
                        n = min(len(pool), rng.randint(*P.facts_per_statement))
                        chosen = [pool.pop() for _ in range(n)]
                        parts, ms = [], []
                        for fid in chosen:
                            para = rng.randrange(3)
                            state[fid].stated.append((tid, cyc, 0, para))
                            parts.append(statement(fid, 0, para))
                            ms.append({"fact_id": fid, "version": 0, "paraphrase": para})
                            ev("stated", cyc, tid, fid, 0, para, via="chat")
                        rec.update(type="state_fact", text=_lead(rng) + " ".join(parts))
                        meta["facts"] = ms
                        kind = "_done"
                elif kind == "ingest_file":
                    n = min(len(pool), rng.randint(*P.file_facts))
                    if n < 3:
                        kind = "filler"
                    else:
                        chosen = [pool.pop() for _ in range(n)]
                        lines, ms = [f"Notes from {_fmt_date(day)}", ""], []
                        for fid in chosen:
                            para = rng.randrange(3)
                            state[fid].stated.append((tid, cyc, 0, para))
                            state[fid].via_file = True
                            lines.append("- " + statement(fid, 0, para))
                            ms.append({"fact_id": fid, "version": 0, "paraphrase": para})
                            ev("stated", cyc, tid, fid, 0, para, via="file")
                        rec.update(type="ingest_file", source="file", file_name=f"notes-{day.isoformat()}-{t}.md",
                                   text="\n".join(lines))
                        meta["facts"] = ms
                        kind = "_done"
                elif kind == "update_fact":
                    cands = [f for f in live if facts[f]["updatable"] and state[f].updated_cycle is None
                             and len(state[f].stated) < 2]  # twice-stated facts are kept for forget requests
                    if not cands:
                        kind = "filler"
                    else:
                        fid = rng.choice(cands)
                        para = rng.randrange(3)
                        s = state[fid]
                        s.version, s.updated_cycle = 1, cyc
                        s.stated.append((tid, cyc, 1, para))
                        rec.update(type="update_fact", text=rng.choice(["Update: ", "Actually, ", "Correction: ", "Things changed. "])
                                   + statement(fid, 1, para))
                        meta["facts"] = [{"fact_id": fid, "version": 1, "paraphrase": para}]
                        ev("updated", cyc, tid, fid, 1, para)
                        kind = "_done"
                elif kind == "style_rule":
                    if not style_pool and not any(s.stated for s in style_state.values()):
                        kind = "filler"
                    else:
                        if style_pool and (rng.random() < P.new_style_share or not any(s.stated for s in style_state.values())):
                            fid = style_pool.pop()
                            para = rng.randrange(3)
                            style_state[fid].stated.append((tid, cyc, 0, para))
                            ev("stated", cyc, tid, fid, 0, para, via="chat")
                        else:
                            fid = rng.choice([f for f, s in style_state.items() if s.stated])
                            first = style_state[fid].stated[0][3]
                            para = rng.choice([p for p in range(3) if p != first])
                            style_state[fid].stated.append((tid, cyc, 0, para))
                            ev("restated", cyc, tid, fid, 0, para)
                        rec.update(type="style_rule", text=_lead(rng) + statement(fid, 0, para))
                        meta["facts"] = [{"fact_id": fid, "version": 0, "paraphrase": para}]
                        kind = "_done"
                elif kind == "tool_task":
                    ti = rng.randrange(N_TOOLS)
                    name = TOOL_SCHEMAS[ti]["name"]
                    text, args = _tool_instance(name, rng)
                    tools_seen.setdefault(name, cyc)
                    rec.update(type="tool_task", text=text, tools=[TOOL_SCHEMAS[ti]])
                    meta.update(tool=name, arguments=args, expected=run_tool(name, args), text=text)
                    kind = "_done"

                if kind == "filler":
                    rec.update(type="filler", text=rng.choice(_FILLER))
                assert "text" in rec, kind
                stream.append(rec)
                turns_meta[tid] = {"type": rec["type"], **meta}

    events.sort(key=lambda e: (e["cycle"], e["turn_id"], e["fact_id"]))
    truth: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "seed": seed, "cycles": cycles, "params": asdict(P),
        "lags": list(LAGS), "sim_start": SIM_START.isoformat(),
        "objects": objects, "facts": facts, "tools": world["tools"],
        "turns": turns_meta, "events": events,
        "forgets": dict(forget_stats),
    }
    truth["questions"] = build_questions(truth, seed)
    return Generated(seed, cycles, stream, truth)


def _do_forget(rec: dict[str, Any], meta: dict[str, Any], state: dict[str, _FactState],
               facts: dict[str, Any], objects: dict[str, Any], fid: str, named: tuple[str, int, int, int],
               others: list[str], twice: bool, cyc: int, tid: str, ev: Callable[..., None]) -> None:
    """A forget request names one entry by its date, the object and the attribute, never the value
    (so the request entry itself holds nothing the byte search would flag)."""
    f = facts[fid]
    named_day = SIM_START + timedelta(days=named[1])
    state[fid].forgotten, state[fid].forgotten_cycle = True, cyc
    text = (f"Please forget what I told you on {_fmt_date(named_day)} about the {f['label']} of "
            f"{objects[f['object_id']]['name']}.")
    rec.update(type="forget_request", text=text)
    meta.update(forget={"fact_id": fid, "named_turn": named[0], "other_turns": others, "stated_twice": twice})
    ev("forgotten", cyc, tid, fid, state[fid].version, -1, named_turn=named[0], other_turns=others,
       stated_twice=twice)


# ---------------------------------------------------------------------------- questions ---


def build_questions(truth: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Evaluation questions (P11 §3), asked after each cycle's consolidation.

    Pure function of ``truth`` (without its ``questions``) and the seed.
    """
    facts, objects, events, cycles = truth["facts"], truth["objects"], truth["events"], truth["cycles"]
    lags = tuple(truth["lags"])
    by_fact: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_fact.setdefault(e["fact_id"], []).append(e)

    def first(fid: str, kind: str) -> dict[str, Any] | None:
        return next((e for e in by_fact.get(fid, []) if e["kind"] == kind), None)

    def cyc_of(e: dict[str, Any] | None) -> int | None:
        return None if e is None else e["cycle"]

    out: list[dict[str, Any]] = []
    counter: dict[int, int] = {}

    def emit(after: int, category: str, text: str, gold: str, fact_ids: list[str], lag: int | None,
             **extra: Any) -> None:
        n = counter.get(after, 0)
        counter[after] = n + 1
        fid = fact_ids[-1]  # the answer's fact (for multi, the port)
        out.append({"qid": f"q{after:04d}.{n:03d}", "after_cycle": after, "category": category, "lag": lag,
                    "text": text, "gold": gold, "fact_ids": fact_ids, "answer_fact_id": fid,
                    "object_id": facts[fid]["object_id"], **extra})

    owners = {f"project:{i:02d}": f"project:{i:02d}.owner" for i in range(N_PROJECTS)}
    for c in range(cycles):
        for fid, f in facts.items():
            cat = f["category"]
            if cat in ("fact", "jargon"):
                st, up, fo = first(fid, "stated"), first(fid, "updated"), first(fid, "forgotten")
                if st is None:
                    continue
                forgotten_by = fo is not None and fo["cycle"] <= c
                updated_by = up is not None and up["cycle"] <= c
                lag = c - st["cycle"]
                if lag in lags and not forgotten_by and not updated_by:
                    emit(c, cat, f["questions"][EVAL_PARAPHRASE], f["versions"][0]["value"], [fid], lag)
                if up is not None and c - up["cycle"] in lags and not forgotten_by:
                    emit(c, "updated", f["questions"][EVAL_PARAPHRASE], f["versions"][1]["value"], [fid],
                         c - up["cycle"], old_value=f["versions"][0]["value"])
                if fo is not None and c - fo["cycle"] in lags:
                    # only never-updated facts can be forgotten (see generate), so the value is v0
                    emit(c, "forgotten", f["questions"][EVAL_PARAPHRASE], f["versions"][0]["value"], [fid],
                         c - fo["cycle"],
                         stated_twice=bool(fo["stated_twice"]), surviving_restatement=bool(fo["other_turns"]))
            elif cat == "style":
                st = first(fid, "stated")
                if st is not None and c - st["cycle"] in lags:
                    r = random.Random(f"neurolearn-style:{seed}:{fid}:{c}")
                    emit(c, "style", r.choice(_STYLE_EVAL_PROMPTS), f["rule"]["kind"], [fid], c - st["cycle"],
                         check=f["rule"])
        for pid, ofid in owners.items():  # multi: project's owner fact and port fact both stated, unchanged
            pfid = pid + ".port"
            so, sp = first(ofid, "stated"), first(pfid, "stated")
            if so is None or sp is None:
                continue
            changed = [first(pfid, "updated"), first(pfid, "forgotten"), first(ofid, "forgotten")]
            if any(e is not None and e["cycle"] <= c for e in changed):
                continue  # from here on the question's gold is no longer the stated port
            lag = c - max(so["cycle"], sp["cycle"])
            if lag in lags:
                person = facts[ofid]["versions"][0]["value"]
                emit(c, "multi", _MULTI_Q[EVAL_PARAPHRASE].format(person=person), facts[pfid]["versions"][0]["value"],
                     [ofid, pfid], lag)
    # tool and anchor questions: every cycle
    tool_first = {}
    for tid, m in truth["turns"].items():
        if m.get("tool"):
            c0 = int(tid[1:5])
            tool_first[m["tool"]] = min(c0, tool_first.get(m["tool"], c0))
    schemas = {s["name"]: s for s in truth["tools"]}
    stream_tool_texts = {m["text"] for m in truth["turns"].values() if m.get("tool")}
    for c in range(cycles):
        for i, (name, c0) in enumerate(sorted(tool_first.items())):
            if c < c0:
                continue
            r = random.Random(f"neurolearn-evaltool:{seed}:{name}:{c}")
            text, args = _tool_instance(name, r)
            while text in stream_tool_texts:  # an evaluation question never equals a stream turn (I-P11-2)
                text, args = _tool_instance(name, r)
            tfid = f"tool:{[s['name'] for s in truth['tools']].index(name)}.use"
            emit(c, "tool", text, run_tool(name, args), [tfid], None,
                 expected_call={"name": name, "arguments": args}, tools=[schemas[name]])
        for k in range(ANCHORS_PER_CYCLE):
            idx = (c * ANCHORS_PER_CYCLE + k) % len(_ANCHOR)
            afid = f"anchor:{idx:02d}.answer"
            emit(c, "anchor", facts[afid]["questions"][0], facts[afid]["versions"][0]["value"], [afid], None)
    return out


# ----------------------------------------------------------------------------- clusters ---


def question_cluster(truth: dict[str, Any], question: dict[str, Any]) -> str:
    """The bootstrap cluster of a question: the object its answer is an attribute of (D-49). A pure
    function of ``truth``; two-hop questions are not chained through intermediate objects."""
    return truth["facts"][question["fact_ids"][-1]]["object_id"]


def cluster_stats(truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per category: questions, clusters and the share of the largest cluster (I-P11-8)."""
    per: dict[str, dict[str, int]] = {}
    for q in truth["questions"]:
        c = per.setdefault(q["category"], {})
        k = question_cluster(truth, q)
        c[k] = c.get(k, 0) + 1
    out = {}
    for cat, counts in sorted(per.items()):
        n = sum(counts.values())
        top = max(counts, key=lambda k: (counts[k], k))
        out[cat] = {"questions": n, "clusters": len(counts), "largest_cluster": top,
                    "largest_share": counts[top] / n}
    return out


# ------------------------------------------------------------------------- oracle rows ---


@dataclass(frozen=True)
class OracleRow:
    """A derived row the oracle supplies for an entry (§2.3) and the turns it must cite."""

    kind: str  # 'qa' | 'implication' | 'probe'
    prompt: str
    answer: str
    source_turns: tuple[str, ...]


def _asof(prefix: str, question: str) -> str:
    return prefix + question[0].lower() + question[1:] if prefix else question


def oracle_rows(gen: Generated, turn_id: str) -> list[OracleRow]:
    """Derived rows for the entry made from ``turn_id`` (inserted with ``origin='oracle'``).

    ``qa`` from question paraphrases 2-3, ``implication`` from the statement paraphrases not used
    in the entry, ``probe`` from the held-out probe paraphrases. A ``multi`` row is added at the
    turn that completes a project's (owner, port) pair, citing both turns.
    """
    truth = gen.truth
    facts = truth["facts"]
    meta = truth["turns"][turn_id]
    rows: list[OracleRow] = []
    turn_by_id = {t["id"]: t for t in gen.stream}
    date_ = SIM_START + timedelta(days=turn_by_id[turn_id]["cycle"])
    for m in meta.get("facts", []):
        f, v, para = facts[m["fact_id"]], m["version"], m["paraphrase"]
        value = f["versions"][v]["value"]
        asof = f"As of {date_.isoformat()}, " if f["time_dependent"] else ""
        if f["category"] == "style":
            rule = f["rule"]
            k = int(f["object_id"].split(":")[1])
            for j, (p, a) in enumerate([_STYLE_TEACH[(2 * k) % 10], _STYLE_TEACH[(2 * k + 1) % 10]]):
                rows.append(OracleRow("qa", p, style_apply(rule, a), (turn_id,)))
            p, a = _STYLE_TEACH[(2 * k + 2) % 10]
            rows.append(OracleRow("probe", p, style_apply(rule, a), (turn_id,)))
        else:
            for qi in ORACLE_PARAPHRASES:
                rows.append(OracleRow("qa", _asof(asof, f["questions"][qi]), value, (turn_id,)))
            for p in f["probes"]:
                rows.append(OracleRow("probe", _asof(asof, p), value, (turn_id,)))
        for si, s in enumerate(f["versions"][v]["statements"]):
            if si != para:
                rows.append(OracleRow("implication", "", s, (turn_id,)))
        # two-hop: owner and port of one project, both in the ledger
        attr, obj = f["attribute"], f["object_id"]
        if obj.startswith("project:") and attr in ("owner", "port"):
            mate = f"{obj}.{'port' if attr == 'owner' else 'owner'}"
            mate_ev = next((e for e in truth["events"] if e["fact_id"] == mate and e["kind"] == "stated"), None)
            this_ev = next((e for e in truth["events"] if e["fact_id"] == f["fact_id"] and e["kind"] == "stated"), None)
            if mate_ev and this_ev and this_ev["turn_id"] == turn_id and mate_ev["turn_id"] <= turn_id \
                    and (mate_ev["turn_id"] != turn_id or attr == "port"):  # same turn: once, at the port
                person = facts[f"{obj}.owner"]["versions"][0]["value"]
                port = facts[f"{obj}.port"]["versions"][0]["value"]
                cite = tuple(sorted({mate_ev["turn_id"], turn_id}))
                for qi in ORACLE_PARAPHRASES:
                    rows.append(OracleRow("qa", _MULTI_Q[qi].format(person=person), port, cite))
    return rows
