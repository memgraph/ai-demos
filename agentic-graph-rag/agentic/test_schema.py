import json
import neo4j


def format_schema():
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))
    with driver.session() as session:
        schema = session.run("SHOW SCHEMA INFO")
        schema_info = json.loads(schema.single().value())
        nodes = schema_info["nodes"]
        edges = schema_info["edges"]
        node_indexes = schema_info["node_indexes"]
        edge_indexes = schema_info["edge_indexes"]

        schema_str = "Nodes:\n"
        for node in nodes:
            properties = ", ".join(
                f"{prop['key']}: {', '.join(t['type'] for t in prop['types'])}"
                for prop in node["properties"]
            )
            schema_str += f"Labels: {node['labels']} | Properties: {properties}\n"

        schema_str += "\nEdges:\n"
        for edge in edges:
            properties = ", ".join(
                f"{prop['key']}: {', '.join(t['type'] for t in prop['types'])}"
                for prop in edge["properties"]
            )
            schema_str += f"Type: {edge['type']} | Start Node Labels: {edge['start_node_labels']} | End Node Labels: {edge['end_node_labels']} | Properties: {properties}\n"

        schema_str += "\nNode Indexes:\n"
        for index in node_indexes:
            schema_str += (
                f"Labels: {index['labels']} | Properties: {index['properties']}\n"
            )

        schema_str += "\nEdge Indexes:\n"
        for index in edge_indexes:
            schema_str += f"Type: {index['type']} | Properties: {index['properties']}\n"

        return schema_str


# Call the function to test it
schema_str = format_schema()
print(schema_str)
