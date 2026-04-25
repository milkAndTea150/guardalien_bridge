"""Codex client adapter for GuardAlien bridge.

The bridge is intentionally runnable in mock mode so your Dify demo does not depend
on Codex availability. To connect your existing Codex bridge, replace
`call_codex_json` with your own implementation or set CODEX_CMD and adapt the
subprocess invocation below.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict
import shlex


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model/CLI response."""
    text = text.strip()
    if not text:
        raise ValueError("Empty Codex response")

    # First try strict JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Then try to find the first JSON object in the output.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from Codex response")


import subprocess
import tempfile
import os
from typing import Dict, Any


def call_codex_json(prompt: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:

        prompt_file = os.path.join(tmpdir, "prompt.txt")

        with open(prompt_file, "w") as f:
            f.write(prompt)

        # 关键：让 codex 在这个目录执行
        cmd = ["codex", "--full-auto", "-m", "gpt-5.4"]  # 🔥 必须

        proc = subprocess.run(
            cmd,
            cwd=tmpdir,  # 🔥 核心：在工作目录运行
            capture_output=True,
            text=True,
            timeout=300,
        )

        # DEBUG
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)

        # 读取生成的文件（你要定义规则）
        return read_generated_project(tmpdir)
