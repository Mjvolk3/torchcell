---
id: cam9jhzi7brp6zk6228dh98
title: Create_scerevisiae_kg
desc: ''
updated: 1712157068371
created: 1705569780034
---

## 2024.03.24 - SSL Apptainer Bug

The code works in docker container but not in my apptainer...

```bash
 [2024-03-24 20:45:21,035][__main__][INFO] - Number of workers: 16
[2024-03-24 20:45:21,062][__main__][INFO] - Instantiating dataset: SmfCostanzo2016Dataset
[2024-03-24 20:45:21,067][__main__][INFO] - Instantiating dataset: SmfKuzmin2018Dataset
[2024-03-24 20:45:21,069][__main__][INFO] - Instantiating dataset: DmfKuzmin2018Dataset
[2024-03-24 20:45:21,071][__main__][INFO] - Instantiating dataset: TmfKuzmin2018Dataset
[2024-03-24 20:45:21,077][__main__][INFO] - Instantiating dataset: DmfCostanzo2016Dataset
[2024-03-24 20:45:21,088][__main__][INFO] - Writing nodes for adapter: SmfCostanzo2016Adapter
Error executing job with overrides: []
Traceback (most recent call last):
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 1298, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 1344, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 1293, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 1052, in _send_output
    self.send(msg)
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 990, in send
    self.connect()
  File "/miniconda/envs/myenv/lib/python3.11/http/client.py", line 1470, in connect
    self.sock = self._context.wrap_socket(self.sock,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/ssl.py", line 517, in wrap_socket
    return self.sslsocket_class._create(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/ssl.py", line 1104, in _create
    self.do_handshake()
  File "/miniconda/envs/myenv/lib/python3.11/ssl.py", line 1382, in do_handshake
    self._sslobj.do_handshake()
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/torchcell/knowledge_graphs/create_scerevisiae_kg.py", line 179, in main
    bc.write_nodes(adapter.get_nodes())
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_core.py", line 277, in write_nodes
    self._get_writer()
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_core.py", line 234, in _get_writer
    translator=self._get_translator(),
               ^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_core.py", line 214, in _get_translator
    ontology=self._get_ontology(),
             ^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_core.py", line 199, in _get_ontology
    self._ontology = Ontology(
                     ^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_ontology.py", line 402, in __init__
    self._main()
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_ontology.py", line 411, in _main
    self._load_ontologies()
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_ontology.py", line 436, in _load_ontologies
    self._head_ontology = OntologyAdapter(
                          ^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_ontology.py", line 97, in __init__
    self._rdf_graph = self._load_rdf_graph(ontology_file)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_ontology.py", line 302, in _load_rdf_graph
    g.parse(ontology_file, format=self._get_format(ontology_file))
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/rdflib/graph.py", line 1470, in parse
    source = create_input_source(
             ^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/rdflib/parser.py", line 416, in create_input_source
    ) = _create_input_source_from_location(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/rdflib/parser.py", line 478, in _create_input_source_from_location
    input_source = URLInputSource(absolute_location, format)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/rdflib/parser.py", line 285, in __init__
    response: addinfourl = _urlopen(req)
                           ^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/site-packages/rdflib/parser.py", line 272, in _urlopen
    return urlopen(req)
           ^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 519, in open
    response = self._open(req, data)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 496, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/miniconda/envs/myenv/lib/python3.11/urllib/request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)>

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
wandb: WARNING No program path found, not creating job artifact. See https://docs.wandb.ai/guides/launch/create-job
```

We fixed this bug with the following:

```python
os.environ["SSL_CERT_FILE"] = certifi.where()
```

## 2024.03.25 - Some Cpus Not Utilized Because How We Set Workers

I think that we just set process workers to max, then threads to some percentage of process workers.

![](./assets/images/torchcell.knowledge_graphs.create_scerevisiae_kg.md.system-cpu-utilization-some-cpus-not-utilized.png)

## 2024.03.28 - Assessing TCDB Build Parameters

[[torchcell.knowledge_graphs.conf.kg.yaml]]

- To get the fastest build `--mem=246g` is the maximum amount of memory that can ever be used on a single `Delta` node. We need the memory, it is the biggest bottleneck. Right now it is difficult to max out cpus without getting an #OOM . Part of the problem is that we need cpus working on larger chunks to get better utilization out of them. Longer chunks means more memory. Since we will always max out memory in our runs and the exchange rate of 1 cpu = 2 g memory, then we claim 123 cpu. Then we can set `io_to_total_worker_ratio` and `process_to_total_worker_ratio`. This will making the comparison of system logs on wandb easier.
- I have neglected the importance of `loader_batch_size` too.
- ⛔️ I was mistaken in thinking that when we wrote `io_workers` that this was referring to threads but in [[Cpu_experiment_loader|dendron://torchcell/torchcell.loader.cpu_experiment_loader]] the workers `arg` is is a `Process` arg. This means we need to revert back to `process_workers = num_workers - io_workers`

| parameter                | description                                                                               |
|:-------------------------|:------------------------------------------------------------------------------------------|
| io_to_total_worker_ratio | Ratio of number of 'process' workers devoted to IO in the loader to total cpu count.      |
| chunk_size               | Chunks from node and edge generators that will be converted to list and eat 🍽️ memory 💾 |
| loader_batch_size        | Size of chunks that the loader will operate on. Makes sense to give                       |

- Makes sense to just auto set batch size on number of workers if we want to simplify but I am not sure if there is room here to explore a bit. Regardless of simplifying, we can try to make chunk_size a multiple of batch as a rough rule of thumb.
- Running 6 hour tests. Should start with parameterization of last successful run.

| experiment | io_to_total_worker_ratio | chunk_size | loader_batch_size | crashed bool | furthest event on `DmfCostanzo2016` | progress ratio |
|:-----------|:-------------------------|:-----------|:------------------|:-------------|:------------------------------------|:---------------|
| 1          | 0.2                      | `1e4`      | `1e3`             |              |                                     |                |
| 2          | 0.4                      | `1e4`      | `1e3`             |              |                                     |                |
| 3          | 0.2                      | `1e4`      | `1e2`             | True         |                                     |                |
| 4          | 0.4                      | `1e4`      | `1e2`             | True         |                                     |                |

- batch size 1e

| experiment | io_to_total_worker_ratio | chunk_size | loader_batch_size | crashed bool | furthest event on `DmfCostanzo2016` | progress ratio |
|:-----------|:-------------------------|:-----------|:------------------|:-------------|:------------------------------------|:---------------|
| 5          | 0.2                      | `1e3`      | `1e2`             |              |                                     |                |
| 6          | 0.2                      | `1e3`      | `1e1`             |              |                                     |                |
| 7          | 0.2                      | `1e2`      | `1e1`             |              |                                     |                |
| 8          | 0.4                      | `1e3`      | `1e2`             |              |                                     |                |
| 9          | 0.2                      | `5e3`      | `1e2`             |              |                                     |                |
| 10         | 0.1                      | `2e3`      | `1e2`             |              |                                     |                |
| 11         | 0.1                      | `8e2`      | `1e2`             |              |                                     |                |
| 12         | 0.8                      | `1e4`      | `1e3`             |              |                                     |                |
| 13         | 0.05                     | `2e3`      | `1e2`             |              |                                     |                |
| 14         | 0.05                     | `2e3`      | `5e2`             |              |                                     |                |
| 15         | 0.05                     | `2e3`      | `1e3`             |              |                                     |                |
| 16         | 0.03                     | `2e3`      | `5e2`             |              |                                     |                |
| 17         | 0.05                     | `2e3`      | `5e2`             |              |                                     |                |

- It is becoming apparent that the most sensitive parameter for controlling memory is `chunk_size`.  
- Moved to #wandb.tcdb.docker_v_m1_study_002
