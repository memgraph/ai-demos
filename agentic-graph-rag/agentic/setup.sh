#!/bin/bash

echo "Starting Memgraph and importing the Ask-news dataset for testing..."

docker run -d --name memgraph_graphRAG -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage:3.0-memgraph-3.0 --log-level=TRACE --also-log-to-stderr --telemetry-enabled=False --schema-info-enabled=True 

sleep 10

# echo "Importing the dataset into Memgraph..."
# cat asknews-finance-graph.cypherl | docker run -i memgraph/mgconsole:latest --host host.docker.internal


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

	