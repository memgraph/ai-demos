import pandas as pd
import random
import datetime
from faker import Faker


fake = Faker()

def generate_iot_infra_graph(num_sensors=50, num_gateways=10, num_power_nodes=20, num_network_nodes=30):
    nodes = []
    edges = []

    # Power Grid Nodes
    for _ in range(num_power_nodes):
        node_id = fake.uuid4()
        nodes.append({
            "id": node_id,
            "type": random.choice(["Substation", "Transformer", "PowerMeter"]),
            "status": random.choice(["OPERATIONAL", "FAULTY"]),
            "location": fake.city()
        })

    # Network Nodes
    for _ in range(num_network_nodes):
        node_id = fake.uuid4()
        nodes.append({
            "id": node_id,
            "type": random.choice(["Router", "Switch", "InternetGateway"]),
            "ip": fake.ipv4(),
            "status": random.choice(["ONLINE", "OFFLINE"])
        })

    # Gateways
    gateways = []
    for _ in range(num_gateways):
        gateway_id = fake.uuid4()
        gateways.append(gateway_id)
        nodes.append({
            "id": gateway_id,
            "type": "Gateway",
            "model": random.choice(["GW-1000", "GW-2000"]),
            "status": random.choice(["ACTIVE", "INACTIVE"])
        })

    # Sensors and Measurements
    start_time = datetime.datetime(2025, 3, 1, 8, 0)
    for _ in range(num_sensors):
        sensor_id = fake.uuid4()
        gateway = random.choice(gateways)
        nodes.append({
            "id": sensor_id,
            "type": random.choice(["Temperature", "Humidity", "Vibration"]),
            "vendor": random.choice(["Bosch", "Siemens"]),
            "location": fake.city(),
            "status": random.choice(["ACTIVE", "INACTIVE"]),
            "gateway": gateway
        })
        
        # Connect Sensor to Gateway
        edges.append((sensor_id, gateway, "CONNECTED_TO"))

        # Generate Measurements
        timestamp = start_time
        for _ in range(100):
            value = round(random.uniform(10, 50), 2)
            anomaly = random.random() < 0.05

            measurement_id = fake.uuid4()
            nodes.append({
                "id": measurement_id,
                "type": "Measurement",
                "value": value,
                "timestamp": timestamp.isoformat(),
                "unit": "°C",
                "sensor": sensor_id,
                "event": "Anomaly" if anomaly else "Normal"
            })
            
            edges.append((sensor_id, measurement_id, "MEASURES"))

            if anomaly:
                event_id = fake.uuid4()
                nodes.append({
                    "id": event_id,
                    "type": "Event",
                    "severity": "HIGH",
                    "description": f"Anomaly detected in {sensor_id}"
                })
                edges.append((measurement_id, event_id, "TRIGGERED"))

            timestamp += datetime.timedelta(minutes=random.randint(5, 15))

    return pd.DataFrame(nodes), pd.DataFrame(edges, columns=["source", "target", "relationship"])

# Generate the data
nodes_df, edges_df = generate_iot_infra_graph()

# Save to CSV
nodes_df.to_csv("iot_nodes.csv", index=False)
edges_df.to_csv("iot_edges.csv", index=False)

print("Dataset generated successfully!")