### Get the whole graph
MATCH path=()-[]-() RETURN path;


### Get the router nodes 
MATCH (n:Router) RETURN n; 


### Get the particular router 
MATCH path=(n:Router{id:9})-[*2]-() RETURN path


### Get the particular controllers 
MATCH (r:Router {id: 9})-[:CONNECTS*]-(c:Controller)
RETURN DISTINCT c

### Get particular end devices 
MATCH (r:Router {id: 9})-[:CONNECTS*]-(c:Controller)-[:HAS]->(d)
RETURN DISTINCT d


### Page rank 
CALL pagerank.get()
YIELD node, rank
RETURN node, rank
ORDER BY rank DESC;