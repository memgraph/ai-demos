import neo4j 



def format_community():
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

    number_of_communities = 0
    with driver.session() as session:
        result = session.run("""
        CALL community_detection.get()
        YIELD node, community_id 
        SET node.community_id = community_id;
        """
        )

        result = session.run("""
        MATCH (n)
        RETURN count(distinct n.community_id) as community_count;
        """
        )
        for record in result:
            number_of_communities = record['community_count']
            print(f"Number of communities: {record['community_count']}")
        

    with driver.session() as session:
        for i in range (0, number_of_communities):
            result = session.run(f"""
            MATCH (start), (end) 
            WHERE start.community_id = {i} AND end.community_id = {i} AND id(start) < id(end)
            MATCH p = (start)-[*..1]-(end)
            RETURN p; 
            """)
        community_string = ""
        for record in result:
            path = record['p']
            for rel in path.relationships:
                start_node = rel.start_node
                end_node = rel.end_node
                start_node_properties = {k: v for k, v in start_node.items() if k != 'embedding'}
                end_node_properties = {k: v for k, v in end_node.items() if k != 'embedding'}
                community_string += f"({start_node_properties})-[:{rel.type}]->({end_node_properties})\n"
        print(community_string)

        

# Call the function to test it
format_community()

        
        