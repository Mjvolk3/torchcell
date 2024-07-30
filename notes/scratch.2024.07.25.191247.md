---
id: ldsex27edlv259hpv3pyt7f
title: '191247'
desc: ''
updated: 1721953235823
created: 1721952769216
---

Probably should have submitted an issue first since path changes can be pretty serious breaking changes and I still don't have a full understanding of the lib. I'll try to clarify why I'd like the change. I think we would probably need to think through additional changes that would be necessary. If you think it is desirable and we can chart out a plan I can go through adding tests, updating docs etc. 

I will reference the [pole](https://github.com/biocypher/pole) project as an example. If you run `pole/create_knowledge_graph.py` it will properly time stamp the directories in the `biocypher-out` dir each of which contains the `neo4j-admin-import-call.sh` which can allow for easy import of the database depending on the dated knowledge graph. The issue I'd like to solve is related to dates imports using the docker container.

`import_call_file_prefix: /data/build2neo` is specified in the `config/biocypher_docker_config.yaml`. This is config is responsible for getting the path right. Inside the container we can look at the import script and see that paths are correct for finding the data.

```bash
root@a3671e4553cd:/var/lib/neo4j/data/build2neo# cat neo4j-admin-import-call.sh
bin/neo4j-admin import --database=neo4j --delimiter="\t" --array-delimiter="|" --quote='+' --force=true --skip-bad-relationships=true --skip-duplicate-nodes=true --nodes="/data/build2neo/Crime-header.csv,/data/build2neo/Crime-part.*" --nodes="/data/build2neo/Officer-header.csv,/data/build2neo/Officer-part.*" --nodes="/data/build2neo/Object-header.csv,/data/build2neo/Object-part.*" --nodes="/data/build2neo/PhoneCall-header.csv,/data/build2neo/PhoneCall-part.*" --nodes="/data/build2neo/Schema_info-header.csv,/data/build2neo/Schema_info-part.*" --nodes="/data/build2neo/Person-header.csv,/data/build2neo/Person-part.*" --nodes="/data/build2neo/Location-header.csv,/data/build2neo/Location-part.*" --relationships="/data/build2neo/INVOLVED_IN-header.csv,/data/build2neo/INVOLVED_IN-part.*" --relationships="/data/build2neo/RECEIVED_CALL-header.csv,/data/build2neo/RECEIVED_CALL-part.*" --relationships="/data/build2neo/IS_RELATED_TO-header.csv,/data/build2neo/IS_RELATED_TO-part.*" --relationships="/data/build2neo/KNOWS-header.csv,/data/build2neo/KNOWS-part.*" --relationships="/data/build2neo/INVESTIGATED_BY-header.csv,/data/build2neo/INVESTIGATED_BY-part.*" --relationships="/data/build2neo/OCCURRED_AT-header.csv,/data/build2neo/OCCURRED_AT-part.*" --relationships="/data/build2neo/LIVES_AT-header.csv,/data/build2neo/LIVES_AT-part.*" --relationships="/data/build2neo/PARTY_TO-header.csv,/data/build2neo/PARTY_TO-part.*" --relationships="/data/build2neo/MADE_CALL-header.csv,/data/build2neo/MADE_CALL-part.*" 

root@a3671e4553cd:/var/lib/neo4j/data/build2neo# ls
Crime-header.csv             IS_RELATED_TO-header.csv   Location-header.csv      Object-header.csv     Person-header.csv          Schema_info-header.csv
Crime-part000.csv            IS_RELATED_TO-part000.csv  Location-part000.csv     Object-part000.csv    Person-part000.csv         Schema_info-part000.csv
INVESTIGATED_BY-header.csv   KNOWS-header.csv           MADE_CALL-header.csv     Officer-header.csv    PhoneCall-header.csv       neo4j-admin-import-call.sh
INVESTIGATED_BY-part000.csv  KNOWS-part000.csv          MADE_CALL-part000.csv    Officer-part000.csv   PhoneCall-part000.csv      schema_info.yaml
INVOLVED_IN-header.csv       LIVES_AT-header.csv        OCCURRED_AT-header.csv   PARTY_TO-header.csv   RECEIVED_CALL-header.csv
INVOLVED_IN-part000.csv      LIVES_AT-part000.csv       OCCURRED_AT-part000.csv  PARTY_TO-part000.csv  RECEIVED_CALL-part000.csv
```

Since this path is fixed the data files are always overwritten in `data/build2neo`. This is the issue I'd like to solve. As pointed out above we have all the date-versioned knowledge graphs in the `biocypher-out` dir and we could just copy these into the container, but then we would also need to make sure that we corrected `neo4j-admin-import-call.sh` as the paths will be wrong. My idea is to keep the data versioning used for both local and the docker container.

With the changes I've described `self._import_call_file_prefix = '/data/build2neo/biocypher-out/20240725174134'`. Then you could separate out builds from import and you could revert to a previous build just by running the import file under the different date-versioned dir. 

My use case is as follows. I am continuing to add new datasets to our current knowledge graph and sometimes the new data breaks things or we update the underlying schema so old queries no longer work. To get the old data out we would like to just redo bulk import of the previously built data then run the query again without having rebuild the csv files since this currently takes hours to days.  

To answer your questions. 

>Your change targets the handling of paths within the neo4j import script, right?

Yes, but it also relates to the `output_directory`. Typically I don't specify this, then I map the `biocypher-out` dir like this `-v "$(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out" \`.

> Isn't this already possible with the current time-stamped output directories? Within the time-stamped directory there is the import script -> versioned by the time-stamp.

I pretty sure this is just for local, but if this can work with docker please let me know.