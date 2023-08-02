---
id: pt6kzbutl4wmnf8xsg4iurb
title: torchcell.tasks
desc: ''
updated: 1690989676878
created: 1690514887023m
---
![[user.mjvolk3.torchcell.tasks.future#future]]

## 2023.08.02

- [ ] 

## 2023.08.01

- [x] Switch to using SGD's genome
- [x] Restructure [[src/torchcell/sgd/sequence.py]] with new data.
- [x] Brainstorm data structure → [[src Drawio|src#src-drawio]]
- [x] Add useful properties [[src/torchcell/sgd/sequence.py]] → I think that [[src/torchcell/sgd/sequence_plot.py]] should be used to split up plotting from `BaseGenome`. *S cerevisiae* is likely to have more data than other organisms so it would be best to make `BaseGenome` as minimal as possible. Could even create a `BaseGenomeSummary` object. We just need to try to force `BaseGenome` to be small. Small enough to contain it's own data, with some extra features for interoperability between the other modules that will compose `Cell`.
- [ ] Read paper on SGD genome update → reference...
- [ ] Add `Zendron`` to repo

## 2023.07.31

- [x] Check images from [[src/torchcell/sgd/validation/raw_structure.py]]
- [x] Work on [[Datasets|src.torchcell.datasets]] charting out a rough map so we don't have to do a ton of rewriting. → update ![](./assets/drawio/Pipeline.drawio.png)
- [x] Change [[src/torchcell/sgd/sgd.py]] from dataclass to attrs. → They are mostly drop in replacement, but `attrs` gives nice funcitonality for comparing objects.
- [ ]
- [ ] Make images SI quality no interfacing with SGD database.

## 2023.07.30

- [x] Graphs of data [[src/torchcell/sgd/validation/raw_structure.py]] → Images look ok.
- [x] Build docs. → Need to add individual modules .rst I think...
- [ ] Add CI for docs, tests, mypy check.
- [ ] Write some pattern for saving files to assets, src, and data. Right now I think the best thing is to copy the images to look the same as the `src`, but I don't think this makes much sense for `data`.
- [ ] Add graph images to note

## 2023.07.29

- [x] Review [[python.lib.pydantic]] models → [[models|python.lib.pydantic.docs.models]]
- [x] Review [[python.lib.pydantic]] field types → [[field types|python.lib.pydantic.docs.field-types]] not yet complete.
- [x] Graphs of data [[src/torchcell/sgd/validation/raw_structure.py]] → They are coming along but still not complete.

## 2023.07.28

- [x] Change the mutability of base model. I think that `BaseDataStrict` should be immutable. → this is shallow immutability, does not act on `dict`

## 2023.07.27

- [x] Process data structure to speed up data validation... will get some nice network plots along the way. → [[src/torchcell/sgd/validation/raw_structure.py]] graphs aren't quite right yet.

## 2023.07.26

- [x] Run a speed test on 10 genes → Very fast comapared to previous yeastmine. this is the way to go instead of yeastmine. [[src/torchcell/sgd.py]]
- [x] Build out pydantic data validation for each of the get data methods... → Started note [[Pydantic|python.lib.pydantic]] to help keep track of best policies and design principles. [[src/torchcell/sgd/validation/locus_related/locus.py]] a few of the test genes pass. Will need to do a more through job of documenting the different types and possibly adding example for the documentation.

## 2023.07.25

- [x] Data → Using SGD API to download data
- [x] Data Async → Sped up with async [[src/torchcell/sgd.py]]
- 🔲 CI setup

## 2023.07.24

- [x] First commit
