# torchcell/knowledge_graphs/generation_image.py
# [[torchcell.knowledge_graphs.generation_image]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/generation_image.py
# Test file: tests/torchcell/knowledge_graphs/test_generation_image.py
"""The prebuilt generation image of the per-adapter KG generation array.

A generation container installs the torchcell commit it builds (``pip install
git+...@BUILD_COMMIT``, then biocypher and CaLM) before the generator starts: 278 s
per container in job 4919_64 (2026-10-15), and the array runs one container per
adapter, 170 of them. This module builds that install ONCE into an image tagged at
the commit, ``tc-neo4j-gen:<first 12 of BUILD_COMMIT>``
(``database/docker/Dockerfile.tc-neo4j-gen``), so a task starts straight into the
generator.

Three rules, enforced here rather than in the shell scripts that call it:

1. **The tag derives from the full commit and from nothing else.** A branch name, a
   short sha or a tag object is refused by ``GenerationImage``, so an image can never
   be named after a moving ref. The image carries the full commit as the label
   ``org.torchcell.build_commit``; ``image_commit`` reads it back and a tag whose
   label is another commit (or none) is a ``GenerationImageMismatch``.
2. **No fallback.** ``ImageMode.IMAGE`` with the tag absent raises
   ``GenerationImageAbsent`` naming the tag and the build command; it never installs
   instead. The install path is selected explicitly (``ImageMode.INSTALL``: the
   serial build, a one-off run). ``ImageMode.AUTO`` is a SELECTION made once, before
   a run, by whoever launches it (image when the tag exists, install otherwise, and
   it says which); the conductor resolves it before the array is submitted, so every
   task runs under an explicit ``image`` or ``install``.
3. **The lifecycle is the conductor's, and the image is outside slurm.** The
   conductor builds (``ensure``) before the arrays are submitted; the live rebuild's
   follow-up removes it after the merge succeeds and keeps it while any task lacks
   its DONE marker (``teardown_decision``), so a re-run of the failed tasks is cheap.
   Docker runs outside the slurm cgroup, so a cancelled or timed-out job leaves the
   image in place; ``remove`` is the by-hand cleanup and ``docker images
   tc-neo4j-gen`` the inventory.

CLI::

    python -m torchcell.knowledge_graphs.generation_image tag     --build-commit <sha>
    python -m torchcell.knowledge_graphs.generation_image check   --build-commit <sha>
    python -m torchcell.knowledge_graphs.generation_image resolve --build-commit <sha> --mode auto
    python -m torchcell.knowledge_graphs.generation_image ensure  --build-commit <sha> [--dry-run]
    python -m torchcell.knowledge_graphs.generation_image build   --build-commit <sha> [--dry-run]
    python -m torchcell.knowledge_graphs.generation_image remove  --build-commit <sha> [--dry-run]

``resolve`` prints ``RESOLVED_MODE=<image|install>`` and ``GEN_IMAGE=<tag>`` for a
shell to read; ``check`` exits 1 with the reason when the tag is absent or carries
another commit; ``ensure`` builds only when the tag is absent and prints
``IMAGE_BUILD_SECONDS=<n>`` (0 when it already existed); ``--dry-run`` prints every
docker command prefixed ``DRY-RUN`` and runs none.
"""

from __future__ import annotations

import argparse
import enum
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

import torchcell

GEN_IMAGE_REPOSITORY = "tc-neo4j-gen"
DEFAULT_BASE_IMAGE = "michaelvolk/tc-neo4j:5.26.28-browser.1"
SHORT_SHA_LENGTH = 12
BUILD_COMMIT_LABEL = "org.torchcell.build_commit"
BASE_IMAGE_LABEL = "org.torchcell.base_image"
DOCKERFILE_RELATIVE = Path("database") / "docker" / "Dockerfile.tc-neo4j-gen"
#: The repository checkout this module was imported from (the Dockerfile lives there).
DEFAULT_TORCHCELL_SRC = Path(torchcell.__file__).resolve().parents[1]

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class ImageMode(enum.StrEnum):
    """How a generation container gets its torchcell: the launcher's choice."""

    IMAGE = "image"
    INSTALL = "install"
    AUTO = "auto"


class ResolvedMode(enum.StrEnum):
    """What a container actually runs under: never ``auto``."""

    IMAGE = "image"
    INSTALL = "install"


class GenerationImageAbsent(RuntimeError):
    """``ImageMode.IMAGE`` was asked for and the tag does not exist. No fallback."""


class GenerationImageMismatch(RuntimeError):
    """The tag exists but its commit label is not the commit asked for."""


class GenerationImage(BaseModel):
    """The image of one build commit: its tag, labels and docker command lines."""

    model_config = ConfigDict(frozen=True)

    build_commit: str
    base_image: str = DEFAULT_BASE_IMAGE

    @field_validator("build_commit")
    @classmethod
    def _full_lowercase_sha(cls, value: str) -> str:
        if not _FULL_SHA.fullmatch(value):
            raise ValueError(
                f"build_commit must be the full 40-hex commit sha, got {value!r} "
                "(a branch, a short sha or a tag object never names an image)"
            )
        return value

    @property
    def tag(self) -> str:
        """``tc-neo4j-gen:<first 12 of the commit>``."""
        return f"{GEN_IMAGE_REPOSITORY}:{self.build_commit[:SHORT_SHA_LENGTH]}"

    @property
    def labels(self) -> dict[str, str]:
        """The labels the Dockerfile stamps: the full commit and the base image."""
        return {
            BUILD_COMMIT_LABEL: self.build_commit,
            BASE_IMAGE_LABEL: self.base_image,
        }

    def dockerfile(self, torchcell_src: Path = DEFAULT_TORCHCELL_SRC) -> Path:
        """The Dockerfile inside the checkout ``torchcell_src``."""
        return torchcell_src / DOCKERFILE_RELATIVE

    def build_argv(self, torchcell_src: Path = DEFAULT_TORCHCELL_SRC) -> list[str]:
        """The ``docker build`` line; the context is the Dockerfile's own directory."""
        dockerfile = self.dockerfile(torchcell_src)
        return [
            "docker",
            "build",
            f"--build-arg=BUILD_COMMIT={self.build_commit}",
            f"--build-arg=BASE_IMAGE={self.base_image}",
            f"--tag={self.tag}",
            f"--file={dockerfile}",
            str(dockerfile.parent),
        ]

    def inspect_argv(self) -> list[str]:
        """``docker image inspect`` printing the commit label alone."""
        return [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{index .Config.Labels " + f'"{BUILD_COMMIT_LABEL}"' + "}}",
            self.tag,
        ]

    def remove_argv(self) -> list[str]:
        """``docker image rm`` of the tag."""
        return ["docker", "image", "rm", self.tag]


def run_docker(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """The default runner: capture output, never raise on a non-zero exit."""
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def image_commit(image: GenerationImage, run: Runner = run_docker) -> str | None:
    """The commit label of ``image.tag``; None when the tag is absent.

    A tag that exists without the label, or with another commit, is a
    ``GenerationImageMismatch``: it was not built by the Dockerfile for this commit.
    """
    result = run(image.inspect_argv())
    if result.returncode != 0:
        return None
    label = result.stdout.strip()
    if label != image.build_commit:
        raise GenerationImageMismatch(
            f"{image.tag} exists but its {BUILD_COMMIT_LABEL} label is "
            f"{label or 'absent'}, not {image.build_commit}; remove it "
            f"({' '.join(image.remove_argv())}) and build again"
        )
    return label


def ensure_present(image: GenerationImage, run: Runner = run_docker) -> None:
    """Raise ``GenerationImageAbsent`` (naming the build command) when the tag is absent."""
    if image_commit(image, run) is None:
        raise GenerationImageAbsent(
            f"{image.tag} is absent for commit {image.build_commit}: no install "
            f"fallback; build it first with "
            f"`python -m torchcell.knowledge_graphs.generation_image build "
            f"--build-commit {image.build_commit}`"
        )


def resolve_mode(
    mode: ImageMode, image: GenerationImage, run: Runner = run_docker
) -> ResolvedMode:
    """The mode a container runs under. ``image`` must exist; ``auto`` is decided here."""
    if mode is ImageMode.INSTALL:
        return ResolvedMode.INSTALL
    if mode is ImageMode.IMAGE:
        ensure_present(image, run)
        return ResolvedMode.IMAGE
    present = image_commit(image, run) is not None
    return ResolvedMode.IMAGE if present else ResolvedMode.INSTALL


def build(
    image: GenerationImage,
    *,
    torchcell_src: Path = DEFAULT_TORCHCELL_SRC,
    dry_run: bool = False,
    run: Runner | None = None,
) -> float:
    """Build the image; return the wall seconds (0.0 for a dry run).

    Without ``run`` the build streams to the caller's stdout/stderr (a docker build is
    minutes of pip output worth seeing in the job log).
    """
    argv = image.build_argv(torchcell_src)
    if dry_run:
        print(f"DRY-RUN {' '.join(argv)}")
        return 0.0
    dockerfile = image.dockerfile(torchcell_src)
    if not dockerfile.is_file():
        raise FileNotFoundError(f"{dockerfile} is not a file")
    t0 = time.monotonic()
    if run is None:
        subprocess.run(argv, check=True)
    else:
        result = run(argv)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, argv, result.stdout, result.stderr
            )
    return time.monotonic() - t0


def ensure(
    image: GenerationImage,
    *,
    torchcell_src: Path = DEFAULT_TORCHCELL_SRC,
    dry_run: bool = False,
    run: Runner = run_docker,
) -> float:
    """Build when the tag is absent; return the build seconds (0.0 when present or dry)."""
    if image_commit(image, run) is not None:
        print(f"{image.tag} present for {image.build_commit}")
        return 0.0
    return build(
        image,
        torchcell_src=torchcell_src,
        dry_run=dry_run,
        run=None if run is run_docker else run,
    )


def remove(
    image: GenerationImage, *, dry_run: bool = False, run: Runner = run_docker
) -> bool:
    """Remove the tag; True when it was removed, False when it was already absent."""
    argv = image.remove_argv()
    if dry_run:
        print(f"DRY-RUN {' '.join(argv)}")
        return False
    if image_commit(image, run) is None:
        print(f"{image.tag} already absent")
        return False
    result = run(argv)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, argv, result.stdout, result.stderr
        )
    print(f"removed {image.tag}")
    return True


class TeardownDecision(BaseModel):
    """What the follow-up does with the image once the array has run."""

    model_config = ConfigDict(frozen=True)

    action: Literal["remove", "keep"]
    reason: str


def teardown_decision(
    *, merge_succeeded: bool, tasks_without_done: int
) -> TeardownDecision:
    """Remove after a successful merge; keep while a re-run of failed tasks is ahead."""
    if tasks_without_done < 0:
        raise ValueError(f"tasks_without_done must be >= 0, got {tasks_without_done}")
    if tasks_without_done > 0:
        return TeardownDecision(
            action="keep",
            reason=f"{tasks_without_done} task(s) lack DONE; the re-run uses the image",
        )
    if not merge_succeeded:
        return TeardownDecision(
            action="keep", reason="the merge did not succeed; the re-run uses the image"
        )
    return TeardownDecision(action="remove", reason="the merge succeeded")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("tag", "print the tag of the commit"),
        ("check", "exit 1 unless the tag exists and carries the commit"),
        ("resolve", "print RESOLVED_MODE= and GEN_IMAGE= for --mode"),
        ("ensure", "build when absent; print IMAGE_BUILD_SECONDS="),
        ("build", "build the image"),
        ("remove", "remove the tag"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--build-commit", required=True, help="the full 40-hex sha")
        p.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
        if name == "resolve":
            p.add_argument(
                "--mode",
                type=ImageMode,
                choices=list(ImageMode),
                default=ImageMode.AUTO,
            )
        if name in ("ensure", "build"):
            p.add_argument("--torchcell-src", type=Path, default=DEFAULT_TORCHCELL_SRC)
        if name in ("ensure", "build", "remove"):
            p.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, run: Runner = run_docker) -> int:
    """The CLI; ``run`` is the docker runner (a fake in the tests)."""
    args = _parser().parse_args(argv)
    image = GenerationImage(build_commit=args.build_commit, base_image=args.base_image)
    try:
        if args.command == "tag":
            print(image.tag)
        elif args.command == "check":
            ensure_present(image, run)
            print(f"{image.tag} present for {image.build_commit}")
        elif args.command == "resolve":
            resolved = resolve_mode(args.mode, image, run)
            print(f"RESOLVED_MODE={resolved.value}")
            print(f"GEN_IMAGE={image.tag}")
        elif args.command == "ensure":
            seconds = ensure(
                image, torchcell_src=args.torchcell_src, dry_run=args.dry_run, run=run
            )
            print(f"IMAGE_BUILD_SECONDS={seconds:.0f}")
            print(f"GEN_IMAGE={image.tag}")
        elif args.command == "build":
            seconds = build(
                image,
                torchcell_src=args.torchcell_src,
                dry_run=args.dry_run,
                run=None if run is run_docker else run,
            )
            print(f"IMAGE_BUILD_SECONDS={seconds:.0f}")
            print(f"GEN_IMAGE={image.tag}")
        elif args.command == "remove":
            remove(image, dry_run=args.dry_run, run=run)
    except (GenerationImageAbsent, GenerationImageMismatch) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
