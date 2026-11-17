$pyFiles = Get-ChildItem -Recurse -Filter *.py -File
$totalFiles = (Get-ChildItem -Recurse -File).Count
$totalLines = 0
$totalSize = 0
foreach ($f in $pyFiles) {
    $lines = (Get-Content $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $totalLines += $lines
    $totalSize += $f.Length
}
$stats = @{
    total_files = $totalFiles
    python_files = $pyFiles.Count
    python_total_lines = $totalLines
    python_total_size_bytes = $totalSize
}
$stats | ConvertTo-Json -Depth 3

