# torchcell/knowledge_graphs/supported_queries/__init__.py
# [[torchcell.knowledge_graphs.supported_queries]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/supported_queries/__init__.py
"""Supported-query registry, Cypher dependency extraction and the drift check.

``registry`` holds the pydantic models and ``registry.json``; ``cypher_deps`` extracts what
a ``.cql`` file reads from the graph; ``check`` compares every registered query with a
committed release snapshot. CLI: ``python -m torchcell.knowledge_graphs.supported_queries
{check,list,validate}``. See [[database.supported-queries]].
"""
