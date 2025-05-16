## Goal

We are writing the experimental pipeline for `Dcell`. It is very similar to the pipeline for the `Dango` model. We want to keep it similar to the Dango pipeline as possible. We want to record all of the same metrics, use the same code structure, etc.

## Relevant Files

These are the relevant Dango files

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dango.py
/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dango_kuzmin2018_tmi.yaml

## Files We Need to Write

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/losses/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_batch_005.py

This configuration file should probably be okay. We might need to make some tweaks to it.

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml

## Update

The pipeline now runs.

## Task

Not all data is on device... We are trying to run on gpu now. Help me make sure everything is on correct device.

***

This was our last attempt
