"""Check that the playback page's script parses (node --check on the inline <script>)."""
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
