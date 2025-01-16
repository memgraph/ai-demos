#!/bin/bash

echo "Starting Memgraph and importing the GOT dataset for testing..."

docker run -d --name memgraph_graphRAG -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage:1.22.1-memgraph-2.22.1 --log-level=TRACE --also-log-to-stderr --telemetry-enabled=False --schema-info-enabled=True --experimental-enabled=vector-search --experimental-config='{ "vector-search": { "index_name": { "label": "Entity", "property": "embedding", "dimension": 128, "capacity": 10000, "metric": "cos", "resize_coefficient": 2}}}'

sleep 10

echo "Importing the dataset into Memgraph..."
cat ../memgraph-export-embeddings-label.cypherl | docker run -i memgraph/mgconsole:latest --host host.docker.internal


# Wait for user to press Ctrl+C
echo "Press Ctrl+C to stop the Memgraph container..."
trap 'echo "Stopping Memgraph container..."; docker stop memgraph_graphRAG; echo "Removing Memgraph container..."; docker rm memgraph_graphRAG; exit' SIGINT

# Keep the script running
while true; do
    sleep 1
done

docker stop memgraph_graphRAG

echo "Removing Memgraph container..."
docker rm memgraph_graphRAG

	