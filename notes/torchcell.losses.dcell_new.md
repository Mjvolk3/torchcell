---
id: p09xcusgkst738dqvr80mcl
title: Dcell_new
desc: ''
updated: 1746936281709
created: 1746936084394
---
## 2025.05.10 - Investigation on Inability to Overfit

```python
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/tor
chcell/models/dcell_new.py
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
Loading sample data with DCellGraphProcessor...
DATA_ROOT: /Users/michaelvolk/Documents/projects/torchcell
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
[2025-05-10 21:47:09,861][torchcell.graph.graph][INFO] - Nodes annotated after 2017-07-19 removed: 2435
After date filter: 3439
[2025-05-10 21:47:09,891][torchcell.graph.graph][INFO] - IGI nodes removed: 160
After IGI filter: 3279
[2025-05-10 21:47:09,956][torchcell.graph.graph][INFO] - Redundant nodes removed: 15
After redundant filter: 3264
[2025-05-10 21:47:10,216][torchcell.graph.graph][INFO] - Nodes with < 2 contained genes removed: 1022
After containment filter: 2242

Normalization parameters for gene_interaction:
  mean: -0.048011
  std: 0.053502
  min: -1.081600
  max: 0.000000
  q25: -0.061951
  q75: -0.015263
  strategy: standard
[2025-05-10 21:47:10,423][torchcell.datamodules.cell][INFO] - Loading index from /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_seed_42.json
[2025-05-10 21:47:10,435][torchcell.datamodules.cell][INFO] - Loading index details from /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Setting up PerturbationSubsetDataModule...
Loading cached index files...
Creating subset datasets...
Setup complete.
  0%|                                                                                                               | 0/1251 [00:00<?, ?it/s]/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/goatools/__init__.py:2: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import get_distribution, DistributionNotFound
  0%|                                                                                                               | 0/1251 [00:17<?, ?it/s]
Using device: cpu
Dataset: 91050 samples, Batch: 32 graphs
Initializing model...
Found 1536 leaf nodes out of 2242 total nodes
Created 2242 subsystems out of 2242 nodes
Total parameters in subsystems: 15,960,612
Created 2242 subsystems from GO graph with 2242 nodes
Initial predictions diversity: 0.384457
✓ Predictions are diverse
Model parameters: 16,010,659
Subsystems: 2,242

Overfitting model on a single batch...
Epoch 5/500, Loss: 0.040379, Corr: 0.0313, Time: 6.770s/epoch
Epoch 10/500, Loss: 0.008510, Corr: 0.2563, Time: 7.270s/epoch
Epoch 15/500, Loss: 0.007070, Corr: 0.1954, Time: 6.780s/epoch
Epoch 20/500, Loss: 0.005411, Corr: 0.1889, Time: 6.619s/epoch
Epoch 25/500, Loss: 0.005123, Corr: 0.3085, Time: 6.588s/epoch
Epoch 30/500, Loss: 0.004344, Corr: 0.3434, Time: 6.478s/epoch
Epoch 35/500, Loss: 0.003913, Corr: 0.4039, Time: 6.522s/epoch
Epoch 40/500, Loss: 0.003614, Corr: 0.3970, Time: 6.826s/epoch
Epoch 45/500, Loss: 0.003294, Corr: 0.4175, Time: 7.082s/epoch
Epoch 50/500, Loss: 0.003039, Corr: 0.4375, Time: 8.610s/epoch
Epoch 55/500, Loss: 0.002825, Corr: 0.4378, Time: 6.572s/epoch
Epoch 60/500, Loss: 0.002631, Corr: 0.4475, Time: 6.738s/epoch
Epoch 65/500, Loss: 0.002463, Corr: 0.4518, Time: 8.174s/epoch
Epoch 70/500, Loss: 0.002321, Corr: 0.4502, Time: 6.796s/epoch
Epoch 75/500, Loss: 0.002191, Corr: 0.4537, Time: 6.712s/epoch
Epoch 80/500, Loss: 0.002078, Corr: 0.4540, Time: 7.619s/epoch
Epoch 85/500, Loss: 0.001979, Corr: 0.4541, Time: 7.883s/epoch
Epoch 90/500, Loss: 0.001893, Corr: 0.4544, Time: 7.773s/epoch
Epoch 95/500, Loss: 0.001818, Corr: 0.4545, Time: 6.596s/epoch
Epoch 100/500, Loss: 0.001753, Corr: 0.4546, Time: 7.362s/epoch
Epoch 105/500, Loss: 0.001697, Corr: 0.4547, Time: 6.480s/epoch
Epoch 110/500, Loss: 0.001650, Corr: 0.4547, Time: 7.246s/epoch
Epoch 115/500, Loss: 0.001609, Corr: 0.4547, Time: 8.032s/epoch
Epoch 120/500, Loss: 0.001574, Corr: 0.4547, Time: 6.814s/epoch
Epoch 125/500, Loss: 0.001545, Corr: 0.4547, Time: 6.840s/epoch
Epoch 130/500, Loss: 0.001520, Corr: 0.4548, Time: 7.578s/epoch
^[[AEpoch 135/500, Loss: 0.001498, Corr: 0.4548, Time: 6.731s/epoch
Epoch 140/500, Loss: 0.001479, Corr: 0.4548, Time: 6.822s/epoch
Epoch 145/500, Loss: 0.001462, Corr: 0.4548, Time: 6.682s/epoch
Epoch 150/500, Loss: 0.001448, Corr: 0.4548, Time: 6.640s/epoch
Epoch 155/500, Loss: 0.001436, Corr: 0.4548, Time: 6.808s/epoch
Epoch 160/500, Loss: 0.001425, Corr: 0.4548, Time: 7.361s/epoch
Epoch 165/500, Loss: 0.001416, Corr: 0.4548, Time: 7.195s/epoch
Epoch 170/500, Loss: 0.001408, Corr: 0.4548, Time: 6.985s/epoch
Epoch 175/500, Loss: 0.001400, Corr: 0.4548, Time: 6.877s/epoch
Epoch 180/500, Loss: 0.001394, Corr: 0.4548, Time: 7.481s/epoch
Epoch 185/500, Loss: 0.001388, Corr: 0.4548, Time: 7.784s/epoch
Epoch 190/500, Loss: 0.001383, Corr: 0.4548, Time: 7.311s/epoch
Epoch 195/500, Loss: 0.001378, Corr: 0.4548, Time: 6.774s/epoch
Epoch 200/500, Loss: 0.001374, Corr: 0.4548, Time: 7.709s/epoch
Epoch 205/500, Loss: 0.001370, Corr: 0.4547, Time: 6.785s/epoch
Epoch 210/500, Loss: 0.001370, Corr: 0.4518, Time: 7.953s/epoch
Epoch 215/500, Loss: 0.001466, Corr: 0.3762, Time: 7.567s/epoch
Epoch 220/500, Loss: 0.001375, Corr: 0.4417, Time: 7.225s/epoch
Epoch 225/500, Loss: 0.001377, Corr: 0.4394, Time: 6.582s/epoch
Epoch 230/500, Loss: 0.001363, Corr: 0.4476, Time: 7.613s/epoch
Epoch 235/500, Loss: 0.001357, Corr: 0.4516, Time: 7.829s/epoch
Epoch 240/500, Loss: 0.001354, Corr: 0.4529, Time: 7.647s/epoch
Epoch 245/500, Loss: 0.001351, Corr: 0.4535, Time: 6.589s/epoch
Epoch 250/500, Loss: 0.001349, Corr: 0.4539, Time: 6.580s/epoch
Epoch 255/500, Loss: 0.001348, Corr: 0.4542, Time: 6.629s/epoch
Epoch 260/500, Loss: 0.001346, Corr: 0.4544, Time: 7.599s/epoch
Epoch 265/500, Loss: 0.001345, Corr: 0.4545, Time: 6.605s/epoch
Epoch 270/500, Loss: 0.001344, Corr: 0.4546, Time: 6.660s/epoch
Epoch 275/500, Loss: 0.001343, Corr: 0.4547, Time: 6.686s/epoch
Epoch 280/500, Loss: 0.001342, Corr: 0.4547, Time: 6.661s/epoch
Epoch 285/500, Loss: 0.001341, Corr: 0.4547, Time: 6.797s/epoch
Epoch 290/500, Loss: 0.001341, Corr: 0.4548, Time: 7.764s/epoch
Epoch 295/500, Loss: 0.001340, Corr: 0.4548, Time: 7.851s/epoch
Epoch 300/500, Loss: 0.001339, Corr: 0.4548, Time: 7.867s/epoch
Epoch 305/500, Loss: 0.001339, Corr: 0.4548, Time: 8.815s/epoch
Epoch 310/500, Loss: 0.001339, Corr: 0.4548, Time: 8.629s/epoch
Epoch 315/500, Loss: 0.001338, Corr: 0.4548, Time: 6.523s/epoch
Epoch 320/500, Loss: 0.001338, Corr: 0.4548, Time: 6.472s/epoch
Epoch 325/500, Loss: 0.001338, Corr: 0.4548, Time: 6.874s/epoch
Epoch 330/500, Loss: 0.001337, Corr: 0.4548, Time: 7.178s/epoch
Epoch 335/500, Loss: 0.001337, Corr: 0.4548, Time: 6.889s/epoch
Epoch 340/500, Loss: 0.001337, Corr: 0.4548, Time: 7.324s/epoch
Epoch 345/500, Loss: 0.001337, Corr: 0.4548, Time: 6.608s/epoch
Epoch 350/500, Loss: 0.001337, Corr: 0.4548, Time: 6.987s/epoch
Epoch 355/500, Loss: 0.001336, Corr: 0.4548, Time: 6.940s/epoch
Epoch 360/500, Loss: 0.001336, Corr: 0.4548, Time: 6.887s/epoch
Epoch 365/500, Loss: 0.001336, Corr: 0.4548, Time: 6.908s/epoch
Epoch 370/500, Loss: 0.001336, Corr: 0.4548, Time: 6.920s/epoch
Epoch 375/500, Loss: 0.001336, Corr: 0.4548, Time: 6.888s/epoch
Epoch 380/500, Loss: 0.001336, Corr: 0.4548, Time: 8.642s/epoch
Epoch 385/500, Loss: 0.001336, Corr: 0.4548, Time: 6.539s/epoch
Epoch 390/500, Loss: 0.001336, Corr: 0.4548, Time: 6.763s/epoch
Epoch 395/500, Loss: 0.001336, Corr: 0.4548, Time: 6.742s/epoch
Epoch 400/500, Loss: 0.001336, Corr: 0.4548, Time: 7.313s/epoch
Epoch 405/500, Loss: 0.001336, Corr: 0.4548, Time: 6.930s/epoch
Epoch 410/500, Loss: 0.001336, Corr: 0.4548, Time: 6.822s/epoch
Epoch 415/500, Loss: 0.001336, Corr: 0.4548, Time: 7.240s/epoch
Epoch 420/500, Loss: 0.001336, Corr: 0.4548, Time: 7.281s/epoch
Epoch 425/500, Loss: 0.001336, Corr: 0.4548, Time: 7.113s/epoch
Epoch 430/500, Loss: 0.001336, Corr: 0.4548, Time: 7.116s/epoch
Epoch 435/500, Loss: 0.001336, Corr: 0.4548, Time: 7.245s/epoch
Epoch 440/500, Loss: 0.001336, Corr: 0.4548, Time: 7.204s/epoch
Epoch 445/500, Loss: 0.001336, Corr: 0.4548, Time: 6.645s/epoch
Epoch 450/500, Loss: 0.001336, Corr: 0.4548, Time: 6.726s/epoch
Epoch 455/500, Loss: 0.001335, Corr: 0.4548, Time: 7.027s/epoch
Epoch 460/500, Loss: 0.001335, Corr: 0.4548, Time: 6.686s/epoch
Epoch 465/500, Loss: 0.001335, Corr: 0.4548, Time: 7.428s/epoch
Epoch 470/500, Loss: 0.001335, Corr: 0.4548, Time: 6.700s/epoch
Epoch 475/500, Loss: 0.001335, Corr: 0.4548, Time: 9.427s/epoch
Epoch 480/500, Loss: 0.001335, Corr: 0.4548, Time: 10.019s/epoch
/Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination_string11_0_to_string12_0.py
Epoch 485/500, Loss: 0.001335, Corr: 0.4548, Time: 7.994s/epoch
Epoch 490/500, Loss: 0.001335, Corr: 0.4548, Time: 6.785s/epoch
Epoch 495/500, Loss: 0.001335, Corr: 0.4548, Time: 6.874s/epoch
Epoch 500/500, Loss: 0.001335, Corr: 0.4548, Time: 7.292s/epoch

Detailed loss components plot saved to 'outputs/dcell_loss_components.png'
Time per epoch plot saved to 'outputs/dcell_time_per_epoch.png'
Predictions vs targets plot saved to 'outputs/dcell_predictions_vs_targets.png'

Final loss values:
  Total Loss: 0.001335
  Primary Loss: 0.000969
  Auxiliary Loss: 0.001222
  Weighted Auxiliary Loss: 0.000367

Time statistics:
  Average time per epoch: 7.151s
  First epoch time: 6.852s
  Last epoch time: 7.292s
  Min epoch time: 6.399s
  Max epoch time: 11.138s

Verifying BatchNorm with small batch...
Single batch forward pass succeeded, shape: torch.Size([1, 1])
```

When we look at outputs/dcell_predictions_vs_targets.png we see that for certain samples we cannot overfit them. We thought that this might be due to sample representations being identical to one another but this doesn't seem to be the case.

`experiments/005-kuzmin2018-tmi/scripts/dcell_batch_005_verify_mutant_state.py`

[[Dcell_batch_005_verify_mutant_state|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dcell_batch_005_verify_mutant_state]]

