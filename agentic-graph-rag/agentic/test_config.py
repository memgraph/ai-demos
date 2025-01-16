import neo4j


def format_config():
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))
    with driver.session() as session:
        config = session.run("SHOW CONFIG")
        config_str = "Configurations:\n"
        for record in config:
            config_str += f"Name: {record['name']} | Default Value: {record['default_value']} | Current Value: {record['current_value']} | Description: {record['description']}\n"
        return config_str


# Call the function to test it
config_str = format_config()
print(config_str)
