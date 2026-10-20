param([string]$SlicerPath = '')
$ErrorActionPreference = 'Stop'
if (-not $SlicerPath) {
    $candidates = @(
        'E:\Program Files (x86)\3D Slicer 5.10.0\Slicer.exe'
        "$env:LOCALAPPDATA\slicer.org\Slicer 5.10.0\Slicer.exe"
    )
    $SlicerPath = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $SlicerPath -or -not (Test-Path -LiteralPath $SlicerPath -PathType Leaf)) {
    throw 'Slicer.exe not found. Run: .\Launch_Slicer_Review.ps1 -SlicerPath "C:\path\Slicer.exe"'
}
$moduleRoot = $PSScriptRoot
$startupScript = Join-Path $moduleRoot 'tools\open_review.py'
if ($moduleRoot.Contains('"') -or $startupScript.Contains('"')) { throw 'Unsupported quote in path' }
$slicerArguments = '--no-splash --additional-module-paths "' + $moduleRoot + '" --python-script "' + $startupScript + '"'
# An interactive Slicer window is the requested tool. No installation/settings changes.
Start-Process -FilePath $SlicerPath -ArgumentList $slicerArguments -WindowStyle Normal
