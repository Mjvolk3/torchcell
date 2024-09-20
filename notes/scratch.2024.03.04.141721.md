---
id: uuc46nd1gzhrhxze34g5r5z
title: '141721'
desc: ''
updated: 1709584156001
created: 1709583443258
---
write a bash script command that takes all the notes md files and it opens those with vscode using `code <filename>` from cli. I want there to options for the number of random files to open.

put it here notes/assets/scripts/random_note.sh

Here is what notes looks like, note that I only want you going one layer deep don't go into any directories.

notes
├── ChatGSE.smf-costanzo2016-demo.md
├── Paper.md
├── assets
├── chatGPT.conversation.conda-env-yaml-output.md
├── cli.ncbi.md
├── database.apptainer.md
├── database.docker.md
├── experiments.costanzo_smd_dmf_dcell.dmf_dcell_slim.md
├── experiments.costanzo_smf_dmf_supervised.dmf_costanzo_deepset_1e5.md
├── experiments.costanzo_smf_dmf_supervised.dmf_costanzo_linear.md
├── experiments.dmf_costanzo_deepset.md
├── experiments.dmf_costanzo_deepset.results.01.md
├── experiments.dmf_costanzo_deepset.results.01.report.md
├── experiments.dmf_fitness_demo.md
├── experiments.S288C_genes_multiple_CDS.md
├── experiments.S288C_stop_start_codons.md
├── meet.Sebastian-Lobentanzer.2023.12.18.md
├── mermaid-filter.err
├── neo4j.delta.md
├── outline.01.md
├── outline.md
├── paper.Abstract.md
├── paper.Conclusion.md
├── paper.Introduction.md
├── paper.Results-and-Discussion.md
├── paper.methods.md
├── paper.outline.dmf-fitness-table.01.md
├── paper.outline.dmf-fitness-table.md
├── paper.outline.md
├── paper.results-and-discussion.1.md
├── paper.results-and-discussion.2.md
├── paper.results-and-discussion.3.md
├── paper.results-and-discussion.4.md
├── paper.results-and-discussion.5.md
├── paper.results-and-discussion.6.md
├── paper.supplementary.md
├── paper.title.md
├── publish.md
├── publish.pdf.md
├── publish.word.md
├── pypi-publish.md
├── python.docs.sphinx.md
├── python.lib.gffutils.md
├── python.lib.mypy.config.md
├── python.lib.mypy.ini.md
├── python.lib.pydantic.docs.field-types.md
├── python.lib.pydantic.docs.models.md
├── python.lib.pydantic.md
├── python.lib.pyreverse.md
├── root.md
├── root.schema.yml
├── scratch.2023.09.22.160524.md
├── scratch.2024.03.04.141721.md
├── sequence.py.md
├── src-to-torchcell-refactor.md
├── src.build.md
├── src.md
├── tags.ChatGPT.md
├── tags.pr.biocypher.import_call_file_prefix.md
├── tags.pr.biocypher.md
├── tags.ramble.md
├── tags.wandb.tags.poc.md
├── tags.wandb.tags.subset.md
├── tests.torchcell.adapters.test_costanzo2016_adapter.md
├── tests.torchcell.adapters.test_kuzmin2018_adapter.md
├── tests.torchcell.graph.test_graph.md
├── tests.torchcell.models.test_fungal_up_down_transformer.md
├── torchcell.adapters.costanzo2016_adapter.md
├── torchcell.adapters.kuzmin2018_adapter.md
├── torchcell.data.data.md
├── torchcell.datamodels.ontology_pydantic.md
├── torchcell.datamodels.pydantic.md
├── torchcell.datamodels.schema.md
├── torchcell.datamodules.DCell.md
├── torchcell.datamodules.cell.md
├── torchcell.dataset_readers.reader.md
├── torchcell.datasets.cell.dbtl_example_data_table.md
├── torchcell.datasets.cell.md
├── torchcell.datasets.cell.supported_data_table.md
├── torchcell.datasets.dcell.md
├── torchcell.datasets.dummy.md
├── torchcell.datasets.embedding.md
├── torchcell.datasets.experiment.md
├── torchcell.datasets.fungal_up_down_transformer.md
├── torchcell.datasets.fungal_utr_transformer.md
├── torchcell.datasets.genome.md
├── torchcell.datasets.md
├── torchcell.datasets.nucleotide_transformer.md
├── torchcell.datasets.ontology.md
├── torchcell.datasets.pronto_ontology.md
├── torchcell.datasets.protT5.md
├── torchcell.datasets.scerevisiae.costanzo2016.md
├── torchcell.datasets.scerevisiae.costanzo2016_deprecated.md
├── torchcell.datasets.scerevisiae.fungal_utr.md
├── torchcell.datasets.scerevisiae.kuzmin2018.md
├── torchcell.datasets.scerevisiae.md
├── torchcell.datasets.scerevisiae.neo_costanzo2016.md
├── torchcell.graph.graph.md
├── torchcell.graph.validation.locus_related.locus.md
├── torchcell.knowledge_graphs.create_pypy_scerevisiae_kg.md
├── torchcell.knowledge_graphs.create_scerevisiae_kg.md
├── torchcell.losses.weighted_mse.md
├── torchcell.md
├── torchcell.models.dcell.md
├── torchcell.models.deep_set.md
├── torchcell.models.deep_set_transformer.md
├── torchcell.models.esm2.md
├── torchcell.models.fungal_up_down_transformer.md
├── torchcell.models.llm.md
├── torchcell.models.mlp.md
├── torchcell.models.protT5.md
├── torchcell.multidigraph.graph.md
├── torchcell.multidigraph.metabolism.md
├── torchcell.multidigraph.sgd.md
├── torchcell.multidigraph.uniprot_api_ec.md
├── torchcell.ncbi.md
├── torchcell.neo4j_fitness_query.md
├── torchcell.ontology.tc_ontology.md
├── torchcell.sequence.data.md
├── torchcell.sequence.genome.scerevisiae.S288C.md
├── torchcell.sequence.genome.scerevisiae.S288C_gb.md
├── torchcell.sequence.genome.scerevisiae.S288C_ncbi.md
├── torchcell.sgd.genome.md
├── torchcell.sgd.sequence.md
├── torchcell.sgd.validation.raw_structure.md
├── torchcell.trainers.graph_convolution_regression.md
├── torchcell.trainers.regression.md
├── torchcell.trainers.regression_deep_set_transformer.md
├── torchcell.viz.fitness.md
├── torchcell.viz.genetic_interaction_score.md
├── user.Heng-Ji.md
├── user.Mjvolk3.torchcell.tasks.md
├── user.liUnderstandingDisharmonyDropout2018.md
├── user.mjvolk3.torchcell.tasks.future.md
├── user.zrimecDeepLearningSuggests2020.md
├── versioning.md
├── vscode.extensions.md
└── workspace.tutorial.md

2 directories, 137 files

```bash
#!/bin/bash

# Path to the notes directory
notes_path="notes"

# Script directory to store the script
script_dir="${notes_path}/assets/scripts"

# Create the scripts directory if it doesn't exist
mkdir -p "$script_dir"

# Script filename
script_file="${script_dir}/random_note.sh"

# Create or overwrite the script file
cat << 'EOF' > "$script_file"
#!/bin/bash

# Check if the number of random files to open is passed as an argument
if [[ $# -eq 0 ]]; then
  echo "Please specify the number of random notes to open."
  exit 1
fi

# Number of files to open
num_files=$1

# Directory containing the note files
note_dir="notes"

# Find all md files in the note directory (without recursing into subdirectories), shuffle them, and take the top N
files_to_open=$(find "$note_dir" -maxdepth 1 -type f -name "*.md" | shuf -n "$num_files")

# Use 'code' command to open each selected file
for file in $files_to_open; do
  code "$file"
done
EOF

# Make the script executable
chmod +x "$script_file"

echo "Script created at: $script_file"
```

The provided script achieves the following:

1. It creates a directory for scripts within your `notes/assets/scripts` path if it doesn't already exist.
2. Inside this directory, it creates a bash script named `random_note.sh`.
3. This script, when executed, requires you to specify the number of random `.md` note files you want to open with VSCode. If no number is provided, it will exit and prompt you to specify the number.
4. It then finds all `.md` files directly under the `notes` directory (no recursion into subdirectories), shuffles this list, and selects the top N files, where N is the number you specified.
5. These files are then opened in VSCode using the `code` command.

To run the script:

1. Navigate to your `notes/assets/scripts` directory.
2. Execute the script by running `./random_note.sh N`, replacing `N` with the number of random note files you wish to open.

This approach uses simple bash commands like `mkdir`, `cat`, `chmod`, `find`, `shuf`, and loops to accomplish the task.

It seems like you've changed your initial request from needing a bash script to wanting the solution in Python using `argparse`. However, the detailed solution you provided is a complete bash script. Could you clarify whether you still need a Python version of this functionality, and if so, could you confirm the exact requirements for the Python script?

The requirements of the python script are to run in the cli with a set of arguments. n for number of notes is a good start. I am thinking running the /ntoes/assets/scripts/rancom_note.py -n 3 or something like that, which would open 3 random md notes in vscode .It appears there was a misunderstanding in your request. If you need further assistance or adjustments to the provided script or a Python version, please let me know!

