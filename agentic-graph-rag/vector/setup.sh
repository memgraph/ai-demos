#!/bin/bash

echo "Starting Memgraph with vector search from custom image for testing on GOT dataset..."
# docker run -d --name memgraph_vector -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage:exp-vector-1 --log-level=DEBUG --also-log-to-stderr --telemetry-enabled=False --experimental-vector-indexes='tag__Entity__embedding__{"dimension":128,"limit":10000}'
docker run -d --name memgraph_vector -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage --log-level=TRACE --also-log-to-stderr --telemetry-enabled=False 
if [ -f "dataset.cypherl.gz" ]; then
    echo "Unzipping the dataset.cypherl.gz file..."
    gunzip dataset.cypherl.gz
else
    echo "dataset.cypherl.gz file not found!"
    echo "Probably the dataset.cypher file is already unzipped."
fi

sleep 3

echo "Importing the dataset into Memgraph..."
cat ../memgraph-export-embeddings-label.cypherl | docker run -i memgraph/mgconsole:latest --host host.docker.internal


# Wait for user to press Ctrl+C
echo "Press Ctrl+C to stop the Memgraph container..."
trap 'echo "Stopping Memgraph container..."; docker stop memgraph_vector; echo "Removing Memgraph container..."; docker rm memgraph_vector; exit' SIGINT

# Keep the script running
while true; do
    sleep 1
done

docker stop memgraph_vector

echo "Removing Memgraph container..."
docker rm memgraph_vector

	
# {
#    "id": 2,
#    "labels": [
#       "Character"
#    ],
#    "properties": {
#       "embedding": [
#          0.8713272213935852,
#          1.1093124151229858
#       ],
#       "name": "Viserys Targaryen"
#    },
#    "type": "node"
# }