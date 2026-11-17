import json
import pathlib
root = pathlib.Path('.')
file_counts = {}
py_lines = 0
py_size = 0
for path in root.rglob('*'):
    if path.is_file():
        ext = path.suffix.lower()
        file_counts[ext] = file_counts.get(ext, 0) + 1
        if ext == '.py':
            try:
                with open(path, encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                py_lines += lines
                py_size += path.stat().st_size
            except Exception:
                pass
metrics = {
    'file_counts': file_counts,
    'python': {
        'files': file_counts.get('.py', 0),
        'lines': py_lines,
        'size_bytes': py_size
    }
}
print(json.dumps(metrics, indent=2))

