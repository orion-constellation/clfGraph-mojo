'''
Memgraph Graph Database:



'''
from memgraph import Memgraph


def store_nodes_in_memgraph(id, memgraph, nodes):
    for node in nodes:
        query = f"""
        CREATE (n:{node['type']} {{
            id: '{node['id']}',
            name: '{node.get('name', '')}',
            created: datetime('{node.get('created', '')}')
        }})
        """
        memgraph.execute(query)

def store_relationships_in_memgraph(id, memgraph, relationships):
    for rel in relationships:
        query = f"""
        MATCH (a {{id: '{rel['source']}'}}), (b {{id: '{rel['target']}'}})
        CREATE (a)-[:{rel['type'].upper()}]->(b)
        """
        memgraph.execute(query)
        