cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(CBCTSegReview)
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
set(MODULE_NAME CBCTSegReview)
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  CBCTSegReviewLib/__init__.py
  CBCTSegReviewLib/core.py
  CBCTSegReviewLib/slicer_adapter.py
  CBCTSegReviewLib/draft.py
  CBCTSegReviewLib/draft_bridge.py
  CBCTSegReviewLib/toothseg.py
)
slicerMacroBuildScriptedModule(NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS})
