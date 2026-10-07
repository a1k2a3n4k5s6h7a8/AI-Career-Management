import os

search_paths = [
    r"c:\Users\HP\Downloads",
    r"c:\Users\HP\OneDrive\Desktop",
    r"c:\Users\HP\OneDrive\Documents"
]

print("Scanning for placement-related datasets...")
for path in search_paths:
    if not os.path.exists(path):
        continue
    print(f"\nChecking path: {path}")
    for root, dirs, files in os.walk(path):
        # Skip hidden or system dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('AppData', 'venv', 'node_modules')]
        for file in files:
            file_lower = file.lower()
            if ('placement' in file_lower or 'student' in file_lower or 'career' in file_lower) and (file_lower.endswith('.csv') or file_lower.endswith('.xlsx')):
                filepath = os.path.join(root, file)
                print(f"Found dataset: {filepath} ({os.path.getsize(filepath)} bytes)")
