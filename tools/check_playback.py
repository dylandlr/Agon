"""Check that the playback page's script parses (node --check on the inline <script>), that the body
model it inlines parses, and that the body model's own tests pass."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

page = Path(__file__).resolve().parent.parent / "scripts" / "playback_template.html"
scripts = re.findall(r"<script>(.*?)</script>", page.read_text(), re.S)
if not scripts:
    sys.exit(f"no inline script in {page}")
with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
    f.write(scripts[-1])
r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
if r.returncode:
    sys.exit(r.stderr)
print(f"{page.name}: script parses ({len(scripts[-1]) // 1024} KB)")

body = page.with_name("body_model.js")
for cmd in (["node", "--check", str(body)], ["node", str(page.parent.parent / "tools" / "test_body_model.js")]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    print(r.stdout.strip() or f"{body.name}: parses")
