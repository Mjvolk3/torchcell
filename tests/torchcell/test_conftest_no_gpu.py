# tests/torchcell/test_conftest_no_gpu
# [[tests.torchcell.test_conftest_no_gpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_conftest_no_gpu
"""The root conftest hides every CUDA device unless ``--gpu`` is given.

Plain ``pytest`` must behave the same on a workstation with four cards as on the CI
runner with none. On 2026-10-02 a full card made seven unmarked tests fail with
``torch.AcceleratorError: CUDA error: out of memory`` because code under test picks
``cuda`` whenever ``torch.cuda.is_available()`` says so. ``tests/conftest.py`` now sets
``CUDA_VISIBLE_DEVICES`` to the empty string with its other environment defaults,
before anything imports torch, which the session and every subprocess inherit.
"""

import os
import subprocess
import sys

import pytest
import torch


def test_plain_session_has_no_cuda_device(request: pytest.FixtureRequest) -> None:
    """Without ``--gpu`` the variable is the empty string and torch counts zero devices."""
    if request.config.getoption("--gpu"):
        pytest.skip("--gpu leaves the devices visible")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert torch.cuda.is_available() is False
    assert torch.cuda.device_count() == 0


def test_a_subprocess_inherits_the_hidden_devices(
    request: pytest.FixtureRequest,
) -> None:
    """A child interpreter started by a test sees the same empty device list."""
    if request.config.getoption("--gpu"):
        pytest.skip("--gpu leaves the devices visible")
    code = (
        "import os, torch; "
        "print(repr(os.environ['CUDA_VISIBLE_DEVICES']), torch.cuda.is_available())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "'' False"
