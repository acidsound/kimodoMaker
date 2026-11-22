"""Added for the second freeze attempt: the local seal must round-trip and tamper must fail."""
import importlib.util
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'v2_gate.py'


def gate():
    spec = importlib.util.spec_from_file_location('v2_gate', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SealTests(unittest.TestCase):
    def test_dpapi_or_local_keystream_roundtrip_and_tamper_detection(self):
        g = gate()
        blob = g.protect(b'sealed test rows')
        if blob[:13] == b'LOCAL-HMAC-1\x00':
            key = g.KEYFILE['seal']
            self.assertEqual(g.keystream_open(blob, key), b'sealed test rows')
            flipped = blob[:14] + bytes([blob[14] ^ 1]) + blob[15:]
            with self.assertRaises(ValueError):
                g.keystream_open(flipped, key)
            with self.assertRaises(ValueError):
                g.keystream_open(b'LOCAL-HMAC-1\x00' + b'0' * 32 + b'x', key)


if __name__ == '__main__':
    unittest.main()
