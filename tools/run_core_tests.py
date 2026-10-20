import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core

buffer = io.StringIO()
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
payload = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run': result.testsRun,
           'failures': len(result.failures), 'errors': len(result.errors), 'log': buffer.getvalue(),
           'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}}
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation' / 'core_tests.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(buffer.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)
