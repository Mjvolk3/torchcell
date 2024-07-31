---
id: eeeb5f654m14lhmt0r85njd
title: Biochatter
desc: ''
updated: 1722383547019
created: 1722369733574
---
## Use Biochatter for Natural Language Query of Database

First export your `OPEN_API_KEY` for LLM querying

```bash
export OPENAI_API_KEY=sk-...
```

```bash
docker run -p 8501:8501 --name app\
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e BIOCHATTER_LIGHT_TITLE="Biochatter" \
  -e BIOCHATTER_LIGHT_HEADER="TorchCell Neo4j Database" \
  -e BIOCHATTER_LIGHT_SUBHEADER="Run Natural Language Queries to get Cypher Query Language Output" \
  -e DOCKER_COMPOSE=false \
  -e CHAT_TAB=false \
  -e PROMPT_ENGINEERING_TAB=false \
  -e RAG_TAB=false \
  -e CORRECTING_AGENT_TAB=false \
  -e KNOWLEDGE_GRAPH_TAB=true \
  -e LAST_WEEKS_SUMMARY_TAB=false \
  -e THIS_WEEKS_TASKS_TAB=false \
  -e TASK_SETTINGS_PANEL_TAB=false \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=torchcell \
  -e NEO4J_AUTH=neo4j/torchcell \
  -e NEO4J_URI=bolt://gilahyper.zapto.org:7687 \
  -e NEO4J_DBNAME=torchcell \
  biocypher/biochatter-light:0.6.12
```

If you get this error you can just remove the container and rerun the `docker run` command.

```bash
docker: Error response from daemon: Conflict. The container name "/app" is already in use by container "e72b55314032f83be2aba57a363e3231073c7c2f2df8fd51e566f535b542ea36". You have to remove (or rename) that container to be able to reuse that name.
See 'docker run --help'.
```

Solution

```bash
docker rm e72b55314032f83be2aba57a363e3231073c7c2f2df8fd51e566f535b542ea36
```

Then try `docker run` again.

The Streamlit app can be found at `http://localhost:8501`
