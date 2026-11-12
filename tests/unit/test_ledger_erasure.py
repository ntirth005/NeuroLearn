"""P01 §4.7 erasure: I-P01-11, ported from reference/sqlite_forget_residue.py."""

from __future__ import annotations

import dataclasses
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from helpers import HashEmbedder, WordTokenizer
from neurolearn.ledger.store import DerivedRow, Ledger, LedgerError

TS = "2026-03-01T10:00:00Z"
CANARIES = ["zqxcanary7731", "Vellmoor"]
SECRET = "My badge code is zqxcanary7731 and I live at 48213 Vellmoor Lane."
WORDS = ["alpha", "beta", "gamma", "delta", "port", "project", "deadline", "owner", "repo", "status"]


def filler(i):
    return " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(30)) + f" item{i}"


def ledger_files(led):
    base = Path(led._db_path)  # noqa: SLF001
    return [base, Path(f"{base}-wal"), Path(f"{base}-shm")]


def residue(led, needles=CANARIES):
    hits = []
    for p, label in zip(ledger_files(led), ("db", "wal", "shm")):
        if p.exists():
            data = p.read_bytes().lower()
            if any(n.lower().encode() in data for n in needles):
                hits.append(label)
    return hits


def plant(led):
    for i in range(150):
        led.append("alice", "chat", filler(i), TS)
    secret = led.append("alice", "chat", SECRET, TS)
    led.add_derived([DerivedRow("qa", "What is the badge code?", "zqxcanary7731")], [secret])
    for i in range(150, 300):
        led.append("alice", "chat", filler(i), TS)
    return secret


@pytest.mark.parametrize("checkpointed_before_wipe", [False, True], ids=["entry-in-wal", "entry-in-db"])
@pytest.mark.parametrize("reader_during_first_checkpoint", [False, True], ids=["no-reader", "reader-snapshot"])
def test_I_P01_11_erase_residue_removes_every_canary_from_every_ledger_file(
        ledger, checkpointed_before_wipe, reader_during_first_checkpoint):
    secret = plant(ledger)
    if checkpointed_before_wipe:
        ledger._conn().raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert residue(ledger), "the canary should be in the files before the wipe"

    ledger.wipe(secret)
    assert residue(ledger), "a logical wipe alone leaves the bytes (05 §3.5)"
    assert ledger.recall("alice", "zqxcanary7731 badge Vellmoor", 5) == [] or all(
        h.entry_id != secret for h in ledger.recall("alice", "zqxcanary7731 badge Vellmoor", 5))

    release = threading.Event()
    reader = None
    if reader_during_first_checkpoint:
        reader = sqlite3.connect(ledger._db_path, isolation_level=None, check_same_thread=False)  # noqa: SLF001
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM entries").fetchone()
        ledger.log("error", {"code": 1})  # a write after the reader's snapshot, as in the reference script

        def hold():
            release.wait(5)
            time.sleep(0.3)
            reader.execute("COMMIT")

        threading.Thread(target=hold).start()
        release.set()

    t0 = time.perf_counter()
    report = ledger.erase_residue(CANARIES)
    waited = time.perf_counter() - t0
    if reader:
        assert waited > 0.2, "the checkpoint should have waited for the reader's snapshot"
        reader.close()

    assert report.checkpoint_ok and report.clean and report.remaining == []
    assert report.steps[:2] == ["checkpoint", "scan"]
    assert residue(ledger) == []
    for c in CANARIES:
        assert c.lower() not in repr(report).lower()
    # the ledger is still usable
    ledger.append("alice", "chat", "a later turn after the forget", TS)
    assert ledger.recall("alice", "later turn forget", 3)


def test_checkpoint_that_cannot_complete_is_reported_not_clean_and_can_be_retried(tmp_path, cfg):
    short = dataclasses.replace(cfg, ledger=dataclasses.replace(cfg.ledger, busy_timeout_ms=200))
    with Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), short) as led:
        secret = plant(led)
        led.wipe(secret)
        reader = sqlite3.connect(led._db_path, isolation_level=None)  # noqa: SLF001
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM entries").fetchone()
        led.log("error", {"code": 1})
        report = led.erase_residue(CANARIES)
        assert not report.checkpoint_ok and not report.clean  # the step has not succeeded until busy = 0
        reader.execute("COMMIT")
        reader.close()
        again = led.erase_residue(CANARIES)
        assert again.checkpoint_ok and again.clean and residue(led) == []


def test_residue_outside_secure_delete_is_removed_by_optimize_and_vacuum(ledger):
    """A copy of the text left in a free page by a writer that did not use secure_delete."""
    ledger.wipe(plant(ledger))
    raw = sqlite3.connect(ledger._db_path, isolation_level=None)  # noqa: SLF001
    raw.execute("PRAGMA secure_delete=OFF")
    raw.execute("BEGIN IMMEDIATE")
    raw.execute("INSERT INTO events(ts, kind, payload) VALUES('t','error',?)", ('{"note":"zqxcanary7731 Vellmoor"}' + "x" * 3000,))
    raw.execute("COMMIT")
    raw.execute("BEGIN IMMEDIATE")
    raw.execute("DELETE FROM events WHERE payload LIKE '%zqxcanary7731%'")
    raw.execute("COMMIT")
    raw.close()
    report = ledger.erase_residue(CANARIES)
    assert report.found_in_ledger_files
    assert "optimize" in report.steps and "vacuum" in report.steps
    assert report.clean and residue(ledger) == []


def test_nothing_to_find_is_clean_straight_away(ledger):
    plant(ledger)
    report = ledger.erase_residue(["never-stored-text-123"])
    assert report.clean and report.steps == ["checkpoint", "scan"] and report.files_scanned >= 1


def test_files_outside_the_ledger_are_deleted_if_derived_artefacts_and_reported_otherwise(ledger):
    root = Path(ledger._dir)  # noqa: SLF001
    (root / "slabs" / "alice").mkdir(parents=True)
    (root / "slabs" / "_originals" / "7").mkdir(parents=True)
    (root / "cache").mkdir()
    (root / "reports" / "run1").mkdir(parents=True)
    (root / "slabs" / "alice" / "s1.v1.safetensors").write_bytes(b"xx zqxcanary7731 xx")
    (root / "slabs" / "alice" / "s2.v1.safetensors.tmp").write_bytes(b"xx ZQXCANARY7731 xx")
    (root / "slabs" / "_originals" / "7" / "old.safetensors").write_bytes(b"VELLMOOR lane")
    (root / "cache" / "logits.bin").write_bytes(b"\x00\x01vellmoor\x02")
    (root / "reports" / "run1" / "stream.jsonl").write_text("zqxcanary7731 is copied here by design")

    report = ledger.erase_residue(CANARIES)

    assert sorted(report.deleted_files) == sorted([
        "slabs/alice/s2.v1.safetensors.tmp", "slabs/_originals/7/old.safetensors", "cache/logits.bin"])
    assert (root / "slabs" / "alice" / "s1.v1.safetensors").exists()  # a live slab file: not ours to delete
    assert (root / "reports" / "run1" / "stream.jsonl").exists()  # reports/ is excluded
    assert report.remaining == [("slabs/alice/s1.v1.safetensors", 0)]
    assert not report.clean
    (root / "slabs" / "alice" / "s1.v1.safetensors").unlink()  # P08 deletes slab files first
    assert ledger.erase_residue(CANARIES).clean


def test_a_canary_straddling_the_read_block_boundary_is_found(ledger):
    root = Path(ledger._dir)  # noqa: SLF001
    (root / "cache").mkdir()
    block = 8 << 20
    needle = b"bridgecanary99"
    with open(root / "cache" / "big.bin", "wb") as f:
        f.write(b"\x00" * (block - 5))
        f.write(needle)
        f.write(b"\x00" * 100)
    report = ledger.erase_residue(["BridgeCanary99"])
    assert report.deleted_files == ["cache/big.bin"] and report.clean


def test_erase_residue_validates_canaries_and_context(ledger):
    with pytest.raises(ValueError):
        ledger.erase_residue([])
    with pytest.raises(ValueError):
        ledger.erase_residue([""])
    with pytest.raises(ValueError, match="1 to 32 bytes"):
        ledger.erase_residue(["x" * 33])
    ledger.erase_residue(["é" * 16])  # 32 bytes exactly
    with pytest.raises(ValueError):
        ledger.erase_residue(["é" * 17])
    with ledger._write():  # noqa: SLF001
        with pytest.raises(LedgerError, match="write transaction"):
            ledger.erase_residue(["abc"])
