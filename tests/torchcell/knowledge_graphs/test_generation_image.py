# tests/torchcell/knowledge_graphs/test_generation_image.py
# [[tests.torchcell.knowledge_graphs.test_generation_image]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_generation_image.py
"""The prebuilt generation image: the tag derives from the full commit only, an
absent tag under ``image`` mode fails loudly with no install fallback, the mode
flag selects image or install, the teardown keeps the image while a re-run is
ahead, and a dry run prints every docker command and runs none. Docker is a fake
runner throughout; the Dockerfile's contract is pinned in its text.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.knowledge_graphs import generation_image as gi
from torchcell.knowledge_graphs.generation_image import (
    BUILD_COMMIT_LABEL,
    DEFAULT_BASE_IMAGE,
    GenerationImage,
    GenerationImageAbsent,
    GenerationImageMismatch,
    ImageMode,
    ResolvedMode,
    build,
    ensure,
    ensure_present,
    image_commit,
    main,
    remove,
    resolve_mode,
    teardown_decision,
)

SHA = "fcb3935c088daafd445676f7e8b35e08eff09924"
OTHER = "9a7d1c2e444a8b279208bb048fc2704dca9fd898"
ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "database" / "docker" / "Dockerfile.tc-neo4j-gen"


class FakeDocker:
    """A docker whose image store is a dict tag -> commit label."""

    def __init__(self, images: dict[str, str] | None = None) -> None:
        """``images`` maps a tag to the commit label it carries."""
        self.images = dict(images or {})
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Answer inspect, build and rm against the dict; refuse anything else."""
        self.calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            tag = argv[-1]
            if tag not in self.images:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"Error: No such image: {tag}\n"
                )
            return subprocess.CompletedProcess(argv, 0, self.images[tag] + "\n", "")
        if argv[:2] == ["docker", "build"]:
            tag = next(a for a in argv if a.startswith("--tag=")).removeprefix("--tag=")
            commit = next(
                a for a in argv if a.startswith("--build-arg=BUILD_COMMIT=")
            ).removeprefix("--build-arg=BUILD_COMMIT=")
            self.images[tag] = commit
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["docker", "image", "rm"]:
            del self.images[argv[-1]]
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected docker call {argv}")

    def commands(self) -> list[str]:
        """The docker verbs called, in order (``inspect``, ``build``, ``rm``)."""
        return [c[1] if c[1] != "image" else c[2] for c in self.calls]


# ------------------------------------------------------------------ tag derivation


def test_the_tag_is_the_repository_and_the_first_twelve_of_the_full_commit() -> None:
    image = GenerationImage(build_commit=SHA)
    assert image.tag == "tc-neo4j-gen:fcb3935c088d"
    assert image.base_image == DEFAULT_BASE_IMAGE
    assert image.labels == {
        BUILD_COMMIT_LABEL: SHA,
        "org.torchcell.base_image": DEFAULT_BASE_IMAGE,
    }


@pytest.mark.parametrize(
    "ref", ["main", "fcb3935c0", SHA.upper(), "v1.7.4", SHA + "0", "origin/main", ""]
)
def test_anything_but_a_full_lowercase_sha_never_names_an_image(ref: str) -> None:
    with pytest.raises(ValidationError, match="full 40-hex commit sha"):
        GenerationImage(build_commit=ref)


def test_two_commits_get_two_tags_and_the_build_line_carries_the_full_commit() -> None:
    a, b = GenerationImage(build_commit=SHA), GenerationImage(build_commit=OTHER)
    assert a.tag != b.tag
    argv = a.build_argv(ROOT)
    assert argv[:2] == ["docker", "build"]
    assert f"--build-arg=BUILD_COMMIT={SHA}" in argv
    assert f"--build-arg=BASE_IMAGE={DEFAULT_BASE_IMAGE}" in argv
    assert f"--tag={a.tag}" in argv
    assert f"--file={DOCKERFILE}" in argv
    assert argv[-1] == str(DOCKERFILE.parent)


# ---------------------------------------------------------- absent tag, no fallback


def test_an_absent_tag_under_image_mode_fails_loudly_and_installs_nothing() -> None:
    docker = FakeDocker()
    image = GenerationImage(build_commit=SHA)
    with pytest.raises(GenerationImageAbsent) as excinfo:
        resolve_mode(ImageMode.IMAGE, image, docker)
    message = str(excinfo.value)
    assert image.tag in message and SHA in message
    assert "no install fallback" in message
    assert f"generation_image build --build-commit {SHA}" in message
    # one inspect, and nothing was built or installed on the way out
    assert docker.commands() == ["inspect"]


def test_check_exits_one_with_the_reason_on_stderr_when_the_tag_is_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    docker = FakeDocker()
    assert main(["check", "--build-commit", SHA], run=docker) == 1
    captured = capsys.readouterr()
    assert "no install fallback" in captured.err
    assert captured.out == ""
    assert docker.commands() == ["inspect"]


def test_a_tag_that_carries_another_commit_is_a_mismatch_not_a_match() -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker({image.tag: OTHER})
    with pytest.raises(GenerationImageMismatch, match=OTHER):
        image_commit(image, docker)
    with pytest.raises(GenerationImageMismatch):
        ensure_present(image, docker)
    unlabeled = FakeDocker({image.tag: ""})
    with pytest.raises(GenerationImageMismatch, match="absent"):
        image_commit(image, unlabeled)


def test_a_present_tag_with_the_right_label_is_found() -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker({image.tag: SHA})
    assert image_commit(image, docker) == SHA
    ensure_present(image, docker)
    assert main(["check", "--build-commit", SHA], run=docker) == 0


# ------------------------------------------------------------------ the mode flag


def test_install_mode_never_touches_docker() -> None:
    docker = FakeDocker()
    assert (
        resolve_mode(ImageMode.INSTALL, GenerationImage(build_commit=SHA), docker)
        is ResolvedMode.INSTALL
    )
    assert docker.calls == []


def test_image_mode_with_the_tag_present_resolves_to_image() -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker({image.tag: SHA})
    assert resolve_mode(ImageMode.IMAGE, image, docker) is ResolvedMode.IMAGE


def test_auto_selects_the_image_when_the_tag_exists_and_install_when_it_does_not() -> (
    None
):
    image = GenerationImage(build_commit=SHA)
    assert resolve_mode(ImageMode.AUTO, image, FakeDocker()) is ResolvedMode.INSTALL
    assert (
        resolve_mode(ImageMode.AUTO, image, FakeDocker({image.tag: SHA}))
        is ResolvedMode.IMAGE
    )


def test_resolve_prints_the_two_lines_a_shell_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = GenerationImage(build_commit=SHA)
    assert (
        main(["resolve", "--build-commit", SHA, "--mode", "auto"], run=FakeDocker())
        == 0
    )
    assert capsys.readouterr().out == f"RESOLVED_MODE=install\nGEN_IMAGE={image.tag}\n"
    assert (
        main(
            ["resolve", "--build-commit", SHA, "--mode", "image"],
            run=FakeDocker({image.tag: SHA}),
        )
        == 0
    )
    assert capsys.readouterr().out == f"RESOLVED_MODE=image\nGEN_IMAGE={image.tag}\n"
    assert (
        main(["resolve", "--build-commit", SHA, "--mode", "image"], run=FakeDocker())
        == 1
    )


def test_the_default_mode_of_resolve_is_auto() -> None:
    assert (
        gi._parser().parse_args(["resolve", "--build-commit", SHA]).mode
        is ImageMode.AUTO
    )


# ------------------------------------------------------------- ensure, build, remove


def test_ensure_builds_only_when_absent_and_the_built_tag_then_checks_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker()
    seconds = ensure(image, torchcell_src=ROOT, run=docker)
    assert seconds >= 0.0
    assert docker.commands() == ["inspect", "build"]
    assert docker.images == {image.tag: SHA}
    assert image_commit(image, docker) == SHA
    capsys.readouterr()
    assert ensure(image, torchcell_src=ROOT, run=docker) == 0.0
    assert docker.commands()[-1] == "inspect"
    assert "present" in capsys.readouterr().out


def test_build_refuses_without_the_dockerfile(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dockerfile.tc-neo4j-gen"):
        build(
            GenerationImage(build_commit=SHA), torchcell_src=tmp_path, run=FakeDocker()
        )


def test_build_raises_on_a_failed_docker_build() -> None:
    def failing(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 2, "", "pip failed")

    with pytest.raises(subprocess.CalledProcessError):
        build(GenerationImage(build_commit=SHA), torchcell_src=ROOT, run=failing)


def test_remove_drops_the_tag_and_is_a_noop_when_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker({image.tag: SHA})
    assert remove(image, run=docker) is True
    assert docker.images == {}
    assert remove(image, run=docker) is False
    assert "already absent" in capsys.readouterr().out


def test_dry_run_prints_the_docker_commands_and_runs_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = GenerationImage(build_commit=SHA)
    docker = FakeDocker()
    assert (
        main(
            [
                "ensure",
                "--build-commit",
                SHA,
                "--torchcell-src",
                str(ROOT),
                "--dry-run",
            ],
            run=docker,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("DRY-RUN docker build ")
    assert f"--build-arg=BUILD_COMMIT={SHA}" in out and f"--tag={image.tag}" in out
    assert "IMAGE_BUILD_SECONDS=0" in out
    assert main(["remove", "--build-commit", SHA, "--dry-run"], run=docker) == 0
    assert capsys.readouterr().out == f"DRY-RUN docker image rm {image.tag}\n"
    # the dry run inspected (to know whether to build) and did nothing else
    assert docker.commands() == ["inspect"]
    assert docker.images == {}


# ------------------------------------------------------------------------ teardown


@pytest.mark.parametrize(
    ("merge_succeeded", "tasks_without_done", "action"),
    [(True, 0, "remove"), (True, 1, "keep"), (False, 0, "keep"), (False, 7, "keep")],
)
def test_the_image_is_removed_only_after_a_clean_merge(
    merge_succeeded: bool, tasks_without_done: int, action: str
) -> None:
    decision = teardown_decision(
        merge_succeeded=merge_succeeded, tasks_without_done=tasks_without_done
    )
    assert decision.action == action
    if action == "keep":
        assert "re-run" in decision.reason


def test_a_negative_task_count_is_refused() -> None:
    with pytest.raises(ValueError):
        teardown_decision(merge_succeeded=True, tasks_without_done=-1)


# --------------------------------------------------------------------- Dockerfile


def test_the_dockerfile_installs_what_the_serial_container_installs_and_verifies_it() -> (
    None
):
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=michaelvolk/tc-neo4j:5.26.28-browser.1" in text
    assert "FROM ${BASE_IMAGE}" in text
    assert "ARG BUILD_COMMIT" in text
    assert 'RUN test -n "$BUILD_COMMIT"' in text
    live = (
        ROOT
        / "database"
        / "slurm"
        / "scripts"
        / "gilahyper_live_rebuild-slurm_docker.slurm"
    ).read_text(encoding="utf-8")
    assert "pip install --no-cache-dir biocypher==0.15.2" in live
    assert "pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main" in live
    assert "--force-reinstall --no-deps" in live
    for line in (
        'python -m pip install --no-cache-dir --force-reinstall --no-deps "git+https://github.com/Mjvolk3/torchcell.git@${BUILD_COMMIT}"',
        "python -m pip install --no-cache-dir biocypher==0.15.2",
        "python -m pip install --no-cache-dir git+https://github.com/oxpig/CaLM@main",
    ):
        assert line in text
    assert 'read_text("direct_url.json")' in text
    assert "is not BUILD_COMMIT" in text
    assert f"LABEL {BUILD_COMMIT_LABEL}=${{BUILD_COMMIT}}" in text
