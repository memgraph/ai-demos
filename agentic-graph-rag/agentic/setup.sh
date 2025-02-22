#!/bin/bash

dataset_path="${1:-asknews-finance-graph.cypherl}"

echo "Starting Memgraph and importing the $dataset_path dataset for testing..."
docker run -d --name memgraph_graphRAG -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage:3.0-memgraph-3.0 --log-level=TRACE --also-log-to-stderr --schema-info-enabled=True 
sleep 10

echo "Importing the dataset into Memgraph..."
lines=$(wc -l < $dataset_path)
batch_size=100
start=1
while [ $start -le $lines ]; do
  sed -n "${start},$((start + batch_size - 1))p" $dataset_path | cat | docker run -i memgraph/mgconsole:latest --host host.docker.internal
  start=$((start + batch_size))
done
# NOTE: cat data | mgconsole doesn't work because mgconsole can take a limited amount/size of queries
# TODO: Fix mgconsole so that it can take file of any size.
# cat "$dataset_path" | docker run -i memgraph/mgconsole:latest --host host.docker.internal

# Wait for user to press Ctrl+C.
echo "Press Ctrl+C to stop the Memgraph container..."
trap 'echo "Stopping Memgraph container..."; docker stop memgraph_graphRAG; echo "Removing Memgraph container..."; docker rm memgraph_graphRAG; exit' SIGINT

# Keep the script running.
while true; do
    sleep 1
done

docker stop memgraph_graphRAG

echo "Removing Memgraph container..."
docker rm memgraph_graphRAG
