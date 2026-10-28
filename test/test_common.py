# Python unit tests for the training/eval helpers (no model download): python -m unittest test/test_common.py
import sys, unittest; sys.path.insert(0, "train")
from common import parse, LABELS, PROMPT
class T(unittest.TestCase):
    def test_parse_exact(self): [self.assertEqual(parse(" " + L), L) for L in LABELS]
    def test_parse_prefix_and_case(self): self.assertEqual(parse(" sports\n"), "Sports"); self.assertEqual(parse(" Tech news"), "Sci/Tech")
    def test_parse_rejects_noise(self): self.assertIsNone(parse(" The headline"))
    def test_prompt_has_all_labels(self): [self.assertIn(L, PROMPT) for L in LABELS]
if __name__ == "__main__": unittest.main()
