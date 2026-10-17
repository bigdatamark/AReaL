# [MODULE DOC]
# Guards against import-time rewriting of PYTORCH_CUDA_ALLOC_CONF. Each check runs
# Python in a subprocess with PYTHONPATH set to the repo root and custom env vars, then
# inspects exit code and output. Also verifies the legacy AWEX_ACTOR_ALLOC_CONF variable
# is no longer referenced and that the awex SGLang plugin refuses expandable segments.
# SPDX-License-Identifier: Apache-2.0
"""Guards that the CUDA allocator config is not rewritten at import time."""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    # [FUNCTION DOC]
    # HELPER: Runs 'python -c code' in a subprocess with the current environment plus
    # PYTHONPATH=REPO_ROOT and env_extra overrides; captures stdout/stderr as text and
    # returns the CompletedProcess. Does not raise on failure.
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


class TestNoImportTimeAllocConfRewrite:
    # [CLASS DOC]
    # Tests that importing areal never mutates the CUDA allocator config env var.
    @pytest.mark.parametrize("role", ["actor", "ref", "rollout"])
    def test_importing_areal_leaves_the_allocator_config_alone(self, role):
        # [FUNCTION DOC]
        # Parametrized over role in actor/ref/rollout: sets sys.argv to include '--role
        # <role>', imports areal with PYTORCH_CUDA_ALLOC_CONF empty and
        # AWEX_ACTOR_ALLOC_CONF set, prints the allocator var as JSON, and asserts the
        # subprocess succeeded and the last stdout line is an empty JSON string.
        code = (
            "import sys, os, json;"
            f" sys.argv = ['prog', '--role', {role!r}];"
            " import areal;"
            " print(json.dumps(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')))"
        )
        result = _run(
            code,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "",
                "AWEX_ACTOR_ALLOC_CONF": "expandable_segments:True",
            },
        )

        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip().splitlines()[-1] == '""', (
            "importing areal rewrote PYTORCH_CUDA_ALLOC_CONF; the per-role "
            "allocator config belongs in each role's scheduling_spec.env_vars"
        )

    def test_awex_actor_alloc_conf_is_gone(self):
        # [FUNCTION DOC]
        # Greps the areal/ source tree (recursively, text files only) for
        # AWEX_ACTOR_ALLOC_CONF and asserts there are no hits.
        hits = subprocess.run(
            ["grep", "-rIl", "AWEX_ACTOR_ALLOC_CONF", "areal"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert hits.stdout.strip() == "", (
            f"AWEX_ACTOR_ALLOC_CONF still referenced in: {hits.stdout.strip()}"
        )


class TestSglangPluginRejectsExpandableSegments:
    # [CLASS DOC]
    # Tests the import-time guard in areal.engine.awex.sglang_plugin.
    def test_import_fails_loudly_when_expandable_segments_is_enabled(self):
        # [FUNCTION DOC]
        # Imports the plugin with expandable_segments:True and asserts a nonzero exit
        # and 'expandable segments' in stderr.
        result = _run(
            "import areal.engine.awex.sglang_plugin",
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

        assert result.returncode != 0
        assert "expandable segments" in result.stderr.lower(), result.stderr[-1500:]

    def test_import_is_allowed_without_expandable_segments(self):
        # [FUNCTION DOC]
        # Imports the plugin with max_split_size_mb:128 and asserts stderr does not
        # mention expandable segments (exit code is not checked, so other import
        # failures are tolerated).
        result = _run(
            "import areal.engine.awex.sglang_plugin",
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        )

        assert "expandable segments" not in result.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
