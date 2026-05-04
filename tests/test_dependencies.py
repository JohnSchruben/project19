from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import unittest


RUN_DEPENDENCY_TESTS = os.environ.get("RUN_DEPENDENCY_TESTS") == "1"


REQUIRED_COMMANDS = (
    "git",
    "uv",
    "docker",
)

REQUIRED_MODULES = (
    ("accelerate", "accelerate"),
    ("av", "av"),
    ("cv2", "opencv-python-headless"),
    ("einops", "einops"),
    ("flash_attn", "flash-attn"),
    ("hydra", "hydra-core"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("physical_ai_av", "physical-ai-av"),
    ("seaborn", "seaborn"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
)


@unittest.skipUnless(RUN_DEPENDENCY_TESTS, "run with --dependencies to check setup dependencies")
class DependencyTests(unittest.TestCase):
    def test_required_commands_are_available(self):
        missing = [command for command in REQUIRED_COMMANDS if shutil.which(command) is None]

        self.assertFalse(
            missing,
            "Missing command-line dependencies: " + ", ".join(missing),
        )

    def test_required_python_modules_are_importable(self):
        missing = [
            package_name
            for import_name, package_name in REQUIRED_MODULES
            if importlib.util.find_spec(import_name) is None
        ]

        self.assertFalse(
            missing,
            "Missing Python packages: " + ", ".join(missing),
        )

    def test_hugging_face_token_is_configured(self):
        token_names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        configured = any(os.environ.get(name) for name in token_names)

        self.assertTrue(
            configured,
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for Alpamayo model downloads.",
        )

    def test_cuda_gpu_is_available(self):
        self.assertIsNotNone(shutil.which("nvidia-smi"), "nvidia-smi is not available.")

        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.strip())

        memory_values = [
            int(value.strip())
            for value in completed.stdout.splitlines()
            if value.strip().isdigit()
        ]
        self.assertTrue(memory_values, "No NVIDIA GPUs were reported by nvidia-smi.")

        minimum_vram_mib = 40 * 1024
        self.assertGreaterEqual(
            max(memory_values),
            minimum_vram_mib,
            "Expected at least one NVIDIA GPU with 40 GB VRAM.",
        )

    def test_python_version_matches_alpamayo(self):
        self.assertEqual(
            sys.version_info[:2],
            (3, 12),
            "Alpamayo pyproject.toml requires Python 3.12.",
        )


if __name__ == "__main__":
    unittest.main()
