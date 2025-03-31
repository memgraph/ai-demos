import random
from time import sleep
from neo4j import GraphDatabase

def generate_and_ingest_iot_graph(uri="bolt://localhost:7687", user="", password="", num_graphs=3, limit=100, ingest=True, filename="iot_graph.cypherl"):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher_statements = []

    # Add indexes
    index_statements = [
        "CREATE INDEX ON :Router(id);",
        "CREATE INDEX ON :AccessPoint(id);",
        "CREATE INDEX ON :WirelessDevice(id);",
        "CREATE INDEX ON :Controller(id);",
        "CREATE INDEX ON :Sensor(id);",
        "CREATE INDEX ON :Actuator(id);",
        "CREATE INDEX ON :Power(id);",
        "CREATE INDEX ON :PowerHub(id);",
        "CREATE INDEX ON :Hub(id);"
    ]
    
    router_device_type = ["Asus", "TP-Link", "Netgear", "Linksys"]
    router_status = ["Online", "Offline"]
    access_point_device_type = ["Cisco", "D-Link", "Ubiquiti"]
    access_point_status = ["Online", "Offline"]
    wireless_device_type = ["Smartphone", "Laptop", "Tablet", "IoTDevice"]
    controller_device_type = ["Zigbee", "Z-Wave", "PLC", "MicroController"]
    controller_status = ["Online", "Offline"]
    sensor_device_type = ["Temperature", "Humidity", "Motion"]
    actuator_device_type = ["Light", "Thermostat", "Lock", "Camera"]
        
    # Counters for each label
    power_id = 0
    router_id = 0
    powerhub_id = 0
    hub_id = 0
    access_point_id = 0
    wireless_device_id = 0
    controller_id = 0
    sensor_id = 0
    actuator_id = 0
        
    # Create power nodes, routers, power hubs
    for _ in range(min(8, limit)):  # Prevent exceeding limit
        power_id += 1
        cypher_statements.append(f"CREATE (:Power {{id: {power_id}}})")
    
    for _ in range(min(10, limit)):  # Prevent exceeding limit
        router_id += 1
        device_type = random.choice(router_device_type)
        status = random.choice(router_status)
        cypher_statements.append(f"CREATE (:Router {{id: {router_id}, device_type: '{device_type}', status: '{status}'}})")
    
    for _ in range(min(3, limit)):  # Prevent exceeding limit
        powerhub_id += 1
        cypher_statements.append(f"CREATE (:PowerHub {{id: {powerhub_id}}})")
        cypher_statements.append(f"MATCH (p:Power) WITH p ORDER BY rand() LIMIT 1 MATCH (ph:PowerHub {{id: {powerhub_id}}}) CREATE (p)-[:SUPPLIES]->(ph)")
    
    # Create hubs, access points, and wireless devices
    for router in range(router_id):
        num_hubs = random.randint(2, 4)
        for _ in range(num_hubs):
            hub_id += 1
            cypher_statements.append(f"CREATE (:Hub {{id: {hub_id}}})")
            cypher_statements.append(f"MATCH (r:Router {{id: {router + 1}}}), (h:Hub {{id: {hub_id}}}) CREATE (r)-[:CONNECTS]->(h)")

            num_access_points = random.randint(1, 3)
            for _ in range(num_access_points):
                access_point_id += 1
                device_type = random.choice(access_point_device_type)
                status = random.choice(access_point_status)
                cypher_statements.append(f"CREATE (:AccessPoint {{id: {access_point_id}, device_type: '{device_type}', status: '{status}'}})")
                cypher_statements.append(f"MATCH (h:Hub {{id: {hub_id}}}), (ap:AccessPoint {{id: {access_point_id}}}) CREATE (ap)-[:CONNECTS]->(h)")

                num_devices = random.randint(1, 3)
                for _ in range(num_devices):
                    wireless_device_id += 1
                    device_type = random.choice(wireless_device_type)
                    cypher_statements.append(f"CREATE (:WirelessDevice {{id: {wireless_device_id}, device_type: '{device_type}'}})")
                    cypher_statements.append(f"MATCH (ap:AccessPoint {{id: {access_point_id}}}), (d:WirelessDevice {{id: {wireless_device_id}}}) CREATE (d)-[:CONNECTS]->(ap)")

        num_controllers = random.randint(2, 3)
        for _ in range(num_controllers):
            controller_id += 1
            device_type = random.choice(controller_device_type)
            status = random.choice(controller_status)
            cypher_statements.append(f"CREATE (:Controller {{id: {controller_id}, device_type: '{device_type}', status: '{status}'}})")
            cypher_statements.append(f"MATCH (h:Hub {{id: {hub_id}}}), (c:Controller {{id: {controller_id}}}) CREATE (c)-[:CONNECTS]->(h)")

            
            num_sensors = random.randint(1, 2)
            for _ in range(num_sensors):
                sensor_id += 1
                device_type = random.choice(sensor_device_type)
                cypher_statements.append(f"CREATE (:Sensor {{id: {sensor_id}, device_type: '{device_type}'}})")
                cypher_statements.append(f"MATCH (c:Controller {{id: {controller_id}}}), (s:Sensor {{id: {sensor_id}}}) CREATE (c)-[:HAS]->(s)")
            
            num_actuators = random.randint(1, 2)
            for _ in range(num_actuators):
                actuator_id += 1
                device_type = random.choice(actuator_device_type)
                cypher_statements.append(f"CREATE (:Actuator {{id: {actuator_id}, device_type: '{device_type}'}})")
                cypher_statements.append(f"MATCH (c:Controller {{id: {controller_id}}}), (a:Actuator {{id: {actuator_id}}}) CREATE (c)-[:HAS]->(a)")
                
            cypher_statements.append(f"MATCH (ph:PowerHub) WITH ph ORDER BY rand() LIMIT 1 MATCH (c:Controller {{id: {controller_id}}}) CREATE (ph)-[:POWERS]->(c)")

    def execute_query(tx, query):
        tx.run(query)

    if ingest:
        # Execute index creation outside of transactions
        with driver.session() as session:
            for index_statement in index_statements:
                session.run("MATCH (n) DETACH DELETE n") 
                sleep(2)
                session.run(index_statement)
                
        for statement in cypher_statements:
            with driver.session() as session:
                session.execute_write(lambda tx: tx.run(statement))
    driver.close()
    print("IoT graph dataset ingested into Memgraph")
        
    with open(filename, "w") as f:
        f.write("\n".join(index_statements))
        f.write("\n")
        f.write("\n".join(cypher_statements))
    
    print(f"Cypher script saved to {filename}")


generate_and_ingest_iot_graph(limit=100, ingest=False)

