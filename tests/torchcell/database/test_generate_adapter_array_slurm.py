# tests/torchcell/database/test_generate_adapter_array_slurm.py
# [[tests.torchcell.database.test_generate_adapter_array_slurm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_generate_adapter_array_slurm.py
"""``gilahyper_generate_adapter_array-slurm_docker.slurm`` and its conductor: the
contract the merge relies on (DONE is written by the generator and verified by the
task, the fence is re-checked per task, the container limits are the tier's), pinned
in the script text; the rest runs docker and sbatch.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "database"
    / "slurm"
    / "scripts"
    / "gilahyper_generate_adapter_array-slurm_docker.slurm"
)
CONDUCTOR = ROOT / "database" / "scripts" / "kg_generation_array_conductor.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
CONDUCTOR_TEXT = CONDUCTOR.read_text(encoding="utf-8")


def test_the_task_is_one_adapter_in_its_own_container_sized_by_the_tier() -> None:
    assert 'INDEX="${SLURM_ARRAY_TASK_ID:?not an array job}"' in TEXT
    assert 'GEN_CONTAINER="tc-neo4j-gen-$INDEX"' in TEXT
    assert (
        '--cpus="$GEN_CPUS" -e SLURM_CPUS_PER_TASK="$GEN_CPUS" --memory="${GEN_MEM_GB}g"'
        in TEXT
    )
    assert "--adapter-index $INDEX --out-dir $OUT_CONT/adapters/$DIRNAME" in TEXT
    assert (
        "python -m torchcell.knowledge_graphs.generate_adapter --config-name $KG_CONFIG"
        in TEXT
    )


def test_the_marker_is_the_generators_last_write_and_the_task_verifies_it() -> None:
    """The script never writes DONE itself; it refuses to end without it, and a
    directory that already carries one is left alone.
    """
    assert (
        "DONE"
        not in TEXT.split("echo GENERATION_DIR")[0].split(
            "docker run --rm --entrypoint bash --name"
        )[1]
    )
    assert (
        '[ -f "$OUT_HOST/adapters/$DIRNAME/DONE" ] || { echo "no DONE marker in $OUT_HOST/adapters/$DIRNAME"; exit 1; }'
        in TEXT
    )
    assert 'if [ -f "$OUT_HOST/adapters/$DIRNAME/DONE" ]; then' in TEXT
    assert "previous incomplete attempt kept at" in TEXT


def test_every_task_rechecks_the_conductors_fence_before_generating() -> None:
    assert (
        "python -m torchcell.knowledge_graphs.generation_array fence --dev-data-root"
        in TEXT
    )
    assert 'if diff "$OUT_HOST/dev_fence.txt" "$FENCE_NOW"; then' in TEXT
    assert "a dev store changed since the conductor digested the tree" in TEXT


def test_the_recovery_command_is_documented_at_the_top() -> None:
    head = TEXT.split("source ~/miniconda3")[0]
    assert "python -m torchcell.knowledge_graphs.generation_array missing" in head
    assert "sbatch --array=<ids>%N" in head
    assert "generation_array merge --out-dir" in head


def test_the_conductor_records_the_order_and_the_fence_and_submits_per_tier() -> None:
    assert (
        "$GA order $PRIVATE_FLAG" in CONDUCTOR_TEXT
        and "adapter_order.json" in CONDUCTOR_TEXT
    )
    assert (
        "$GA fence --dev-data-root" in CONDUCTOR_TEXT
        and "dev_fence.txt" in CONDUCTOR_TEXT
    )
    assert "$GA submit --out-name" in CONDUCTOR_TEXT
    assert 'DRY_RUN="${DRY_RUN:-0}"' in CONDUCTOR_TEXT
    assert "generation_tiers.yaml" in CONDUCTOR_TEXT
