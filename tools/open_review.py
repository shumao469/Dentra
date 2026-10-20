import slicer

slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.util.selectModule('CBCTSegReview')
slicer.util.mainWindow().show()
