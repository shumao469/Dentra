"""Test discovery via --additional-module-paths, including actual Qt module setup."""
import json
from pathlib import Path
import traceback
import slicer

root = Path(__file__).resolve().parents[1]
try:
    # Module selection needs a main window; run this test without --no-main-window.
    slicer.util.mainWindow().hide()
    slicer.util.selectModule('CBCTSegReview')
    widget = slicer.modules.cbctsegreview.widgetRepresentation().self()
    assert widget.editor and widget.session is None
    from CBCTSegReviewLib import core
    payload = {'status': 'PASS', 'slicer_version': slicer.app.applicationVersion,
               'module_registered': True, 'widget_created': True,
               'main_window_module_selection': True,
               'tested_source_sha256': {str(p.relative_to(root)): core.digest_file(p) for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}}
    code = 0
except Exception:
    payload = {'status': 'FAIL', 'traceback': traceback.format_exc()}
    code = 1
(root / 'validation').mkdir(exist_ok=True)
(root / 'validation' / 'slicer_registration.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
slicer.app.exit(code)
