"""CPU-only contract tests for the project-authored v1 corpus."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT / "v1_corpus.py"
sys.path.insert(0, str(EXPERIMENT))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CorpusTests(unittest.TestCase):
    def test_train_output_is_reproducible_and_has_rowwise_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / x for x in ("a", "b"))
            for out in (a, b):
                result = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertEqual({p.name for p in a.iterdir()}, {"train.jsonl", "train_provenance.jsonl", "manifest.json"})
            for p in a.iterdir():
                self.assertEqual(p.read_bytes(), (b / p.name).read_bytes())
            rows = load_jsonl(a / "train.jsonl")
            provenance = load_jsonl(a / "train_provenance.jsonl")
            self.assertTrue(700 <= len(rows) <= 2000)
            self.assertEqual([r["id"] for r in rows], [p["id"] for p in provenance])
            self.assertEqual({frozenset(r) for r in rows}, {frozenset({"id", "group_id", "split", "en", "ko"})})
            for lang in ("en", "ko"):
                self.assertEqual(len({r[lang] for r in rows}), len(rows))
            for key in ("train.jsonl", "train_provenance.jsonl"):
                manifest = json.loads((a / "manifest.json").read_bytes())
                self.assertEqual(manifest["files"][key]["sha256"], hashlib.sha256((a / key).read_bytes()).hexdigest())

    def test_evaluation_sets_are_balanced_and_independently_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public, sealed, sealed2 = [root / x for x in ("public", "sealed", "sealed2")]
            for args in (("--stage", "train-dev", "--out", str(public)),
                         ("--stage", "test", "--out", str(sealed), "--public-out", str(public)),
                         ("--stage", "test", "--out", str(sealed2), "--public-out", str(public))):
                result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertFalse((public / "test.jsonl").exists())
            self.assertNotIn("test.jsonl", (public / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({f.name for f in sealed.iterdir()}, {"test.jsonl", "manifest.json"})
            for p in sealed.iterdir():
                self.assertEqual(p.read_bytes(), (sealed2 / p.name).read_bytes())
            groups = []
            for directory, split in ((public, "dev"), (sealed, "test")):
                path = directory / f"{split}.jsonl"
                rows = load_jsonl(path)
                manifest = json.loads((directory / "manifest.json").read_bytes())
                self.assertEqual(manifest["files"][path.name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(manifest["rules"]["seeds"], [17, 43, 97])
                self.assertEqual(collections.Counter(r["kind"] for r in rows), dict.fromkeys(range(1, 5), 16))
                pair_map = collections.defaultdict(list)
                for row in rows:
                    pair_map[row["group_id"]].append(row)
                    self.assertEqual(row["split"], split)
                    self.assertEqual(set(row), {"id", "group_id", "split", "en", "ko", "kind", "seconds", "unseen", "footing", "intent"})
                    self.assertIn(row["intent"]["type"], ("hand", "turn", "order", "height"))
                    self.assertTrue(2.5 <= row["seconds"] <= 3.5)
                    self.assertIs(type(row["unseen"]), bool)
                    self.assertIs(type(row["footing"]), bool)
                self.assertTrue(any(r["unseen"] for r in rows))
                self.assertEqual(len(pair_map), 32)
                self.assertEqual({len(pair) for pair in pair_map.values()}, {2})
                for a, b in pair_map.values():
                    for key in ("seconds", "unseen", "footing", "kind"):
                        self.assertEqual(a[key], b[key])
                    self.assertNotEqual(a["intent"], b["intent"])
                    if a["kind"] == 3:
                        self.assertEqual((a["intent"]["first"], a["intent"]["second"]), (b["intent"]["second"], b["intent"]["first"]))
                groups.append(set(pair_map))
            self.assertFalse(groups[0] & groups[1])
            for args in (("--stage", "test", "--out", str(public / "sealed"), "--public-out", str(public)),
                         ("--stage", "test", "--out", str(sealed), "--public-out", str(public)),
                         ("--stage", "test", "--out", str(root / "ungated"))):
                result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_audit_detects_real_bilingual_cue_and_split_corruption(self):
        import v1_corpus as corpus

        train, _ = corpus.build_train()
        sets = {"train": train, "dev": corpus.build_evaluation("dev"), "test": corpus.build_evaluation("test")}
        report = corpus.audit(sets)
        self.assertEqual(report["cross_split_overlaps"], {"group_ids": 0, "normalized_en": 0, "normalized_ko": 0, "signed_signatures": 0, "equivalence_signatures": 0})
        for lang in ("en", "ko"):
            self.assertGreater(report["cue_checks"][lang]["ordered_sequence"], 0)
        for split in ("dev", "test"):
            self.assertGreater(report["splits"][split]["unseen_type_combinations"], 0)
            self.assertGreater(report["splits"][split]["known_atom_novel_combinations"], 0)

        mutations = {"en": ((1, "left hand", "right hand"), (2, "to your left", "to your right"),
                            (2, "90 degrees", "180 degrees"), (4, "lower", "rise"),
                            (3, "Then ", "Meanwhile, "), (None, "Slowly ", "Quickly ")),
                     "ko": ((1, "왼손", "오른손"), (2, "왼쪽으로", "오른쪽으로"),
                            (2, "90도", "180도"), (4, "낮추", "일어서"),
                            (3, "그런 다음 ", "동시에 "), (None, "천천히", "빠르게"))}
        for lang, changes in mutations.items():
            for kind, old, new in changes:
                with self.subTest(lang=lang, cue=old):
                    original = next(e for entries in sets.values() for e in entries
                                    if e.row.get("kind") == kind and old in e.row[lang] and (kind is not None or len(e.events) == 1))
                    changed = copy.deepcopy(original)
                    changed.row[lang] = changed.row[lang].replace(old, new, 1)
                    with self.assertRaises(ValueError):
                        corpus.audit_cues(changed, lang)
        for lang in ("en", "ko"):
            original = next(e for e in sets["dev"] if e.row["kind"] == 3 and ("Then " if lang == "en" else "그런 다음 ") in e.row[lang])
            changed = copy.deepcopy(original)
            other = next(e for e in sets["dev"] if e.row["group_id"] == original.row["group_id"] and e.row["id"] != original.row["id"])
            changed.row[lang] = other.row[lang]
            with self.assertRaises(ValueError):
                corpus.audit_cues(changed, lang)

        records = {key: list(value) for key, value in sets.items()}
        mirrored_or_reversed_source = copy.deepcopy(records["dev"][0])
        mirrored_or_reversed_source.row.update(id="foreign-author-id", group_id="foreign-author-group", split="train")
        records["train"].append(mirrored_or_reversed_source)
        with self.assertRaisesRegex(ValueError, "overlap|leakage"):
            corpus.audit(records)
        for lang, transform in (("en", lambda s: s.upper().replace(" ", "  ") + "!"),
                                ("ko", lambda s: s.replace(" ", "\u3000") + "…")):
            with self.subTest(normalization=lang):
                duplicate = copy.deepcopy(train[-1])
                duplicate.row[lang] = transform(train[0].row[lang])
                result = [*train[:-1], duplicate]
                with self.assertRaisesRegex(ValueError, "normalized"):
                    corpus.audit({"train": result})

    def test_public_stage_does_not_read_evaluation_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standalone = root / SCRIPT.name
            standalone.write_bytes(SCRIPT.read_bytes())
            command = [sys.executable, str(standalone)]
            result = subprocess.run([*command, "--out", str(root / "train")], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            dev_source = "v1_corpus_dev.json"
            (root / dev_source).write_bytes((EXPERIMENT / dev_source).read_bytes())
            result = subprocess.run([*command, "--stage", "dev", "--out", str(root / "dev")], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertFalse((root / "dev" / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
