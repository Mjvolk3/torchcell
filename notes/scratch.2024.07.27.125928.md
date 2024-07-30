---
id: ci0o3rfkix3mmkgqa6fjzeh
title: '125928'
desc: ''
updated: 1722103576811
created: 1722103170424
---
```bash
michaelvolk@M1-MV torchcell % docker run -p 8501:8501 --name app \                                                                   12:55
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e DOCKER_COMPOSE=false \
  -e CHAT_TAB=false \
  -e PROMPT_ENGINEERING_TAB=false \
  -e RAG_TAB=false \
  -e CORRECTING_AGENT_TAB=false \
  -e KNOWLEDGE_GRAPH_TAB=true \
  -e NEO4J_URI=bolt://gilahyper.zapto.org:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=torchcell \
  -e NEO4J_AUTH=neo4j/torchcell \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
  biocypher/biochatter-light:0.4.7


Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


  You can now view your Streamlit app in your browser.

  Network URL: http://172.17.0.2:8501
  External URL: http://169.197.145.193:8501
```

```bash
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[2024-07-27 17:56:22.972] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:56:22.974] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:22.974] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:22.974] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:22.974] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:22.974] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:22.974] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:22.975] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:22.975] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:22.975] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:22.975] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:22.975] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.53 ] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:56:23.54 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:23.55 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.56 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:23.58 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.68 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:23.70 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.72 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:23.73 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.74 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:23.75 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:23.76 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connection refused)
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connection refused)
[2024-07-27 17:56:45.359] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:56:45.410] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:56:45.417] [neo4ju.query]                 [ERROR]    Failed to run query: AuthError: {code: Neo.ClientError.Security.Unauthorized} {message: The client is unauthorized due to authentication failure.}
[2024-07-27 17:56:45.417] [neo4ju.query]                 [ERROR]    Authentication error, switching to offline mode.
[2024-07-27 17:56:45.418] [neo4ju.go_offline]            [WARNING]  Offline mode: any interaction to the server is disabled.
[2024-07-27 17:58:49.99 ] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:58:49.140] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:58:49.147] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:58:49.158] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:58:49.170] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:58:49.182] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:58:49.190] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:58:49.202] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:58:49.209] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:58:49.220] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:58:49.226] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:58:49.233] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.890] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:11.904] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:11.922] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.931] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:11.939] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.948] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:11.955] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.962] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:11.969] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.976] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.982] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:11.988] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:11.992] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:11.999] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:12.5  ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:12.12 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:12.24 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:12.42 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:12.50 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:12.56 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:12.63 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:12.68 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:12.74 ] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:12.80 ] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.512] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:14.533] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.543] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.550] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.559] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.567] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.574] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.581] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.588] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.594] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.601] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.607] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.611] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:14.618] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.625] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.631] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.637] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.644] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.650] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.656] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.663] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.669] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:14.675] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:14.681] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.689] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:15.702] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.716] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.723] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.732] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.738] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.746] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.753] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.759] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.765] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.772] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.781] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.785] [neo4ju.read_config]           [WARNING]  No config available, falling back to defaults.
[2024-07-27 17:59:15.791] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.799] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.805] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.812] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
[2024-07-27 17:59:15.818] [neo4ju.query]                 [WARNING]  Running query against fallback database `neo4j`.
[2024-07-27 17:59:15.824] [neo4ju.query]                 [ERROR]    Failed to run query: ServiceUnavailable: The client has provided incorrect authentication details too many times in a row.
```