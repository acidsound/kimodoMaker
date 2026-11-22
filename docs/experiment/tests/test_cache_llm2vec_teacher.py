import importlib.util
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile


SOURCE = Path(__file__).resolve().parents[1] / "cache_llm2vec_teacher.py"


class CacheTeacherTest(unittest.TestCase):
    def test_export_requires_unique_train_pairs(self):
        spec = importlib.util.spec_from_file_location("cache_llm2vec_teacher", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sample = '{"id":"a","group_id":"mirror-group","split":"train","en":"Turn left","ko":"왼쪽으로 돈다"}'
        self.assertEqual(module.train_rows(sample.encode())[0]["group_id"], "mirror-group")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.train_rows((sample + "\n" + sample).encode())
        with self.assertRaisesRegex(ValueError, "train"):
            module.train_rows(sample.replace('"train"', '"test"').encode())

    def test_app_punctuation_and_official_sanitizer(self):
        spec = importlib.util.spec_from_file_location("cache_llm2vec_teacher", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.teacher_prompts([{"en": "  TURN    left "}, {"en": "-raise one hand!"}]),
                         ["Turn left.", "Raise one hand!"])

    def test_upload_zip_only_contains_three_input_files(self):
        spec = importlib.util.spec_from_file_location("cache_llm2vec_teacher", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {"train.jsonl", "targets.npz", "teacher.json"}
            for name in expected | {"test.jsonl"}:
                (root / name).write_bytes(name.encode())
            archive_path = module.write_upload_zip(root)
            self.assertEqual(archive_path.name, "kimodo-train-input.zip")
            with ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), expected)
                for name in expected:
                    self.assertEqual(archive.read(name), (root / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
