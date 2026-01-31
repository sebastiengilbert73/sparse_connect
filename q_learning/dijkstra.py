import pandas as pd
import heapq
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def load_graph(nodes_file, edges_file):
    """
    Loads nodes and edges from CSV files and builds an adjacency list.
    """
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    graph = {node_id: [] for node_id in nodes_df['Id']}
    
    for _, row in edges_df.iterrows():
        source = int(row['Source'])
        target = int(row['Target'])
        weight = float(row['Weight'])
        graph[source].append((target, weight))
        
    return graph

def dijkstra(graph, start_node, end_node):
    """
    Implements Dijkstra's algorithm to find the shortest path between start_node and end_node.
    """
    if start_node not in graph or end_node not in graph:
        return None, float('inf')
    
    # Priority queue stores (cost, current_node, path)
    pq = [(0, start_node, [start_node])]
    visited = set()
    
    while pq:
        (cost, current, path) = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end_node:
            return path, cost
            
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
                
    return None, float('inf')

def main(start,
        end,
        nodes,
        edges):
    
    logging.info(f"dijkstra.main(): Loading graph from {nodes} and {edges}")
    
    # Ensure relative paths are handled correctly if run from project root
    nodes_path = nodes
    edges_path = edges
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        logging.error(f"dijkstra.main(): Could not find files {nodes_path} or {edges_path}")
        return

    graph = load_graph(nodes_path, edges_path)
    path, cost = dijkstra(graph, start, end)
    logging.info(f"dijkstra.main(): path = {path}")
    logging.info(f"dijkstra.main(): cost = {cost}")
    if path:
        logging.info(f"dijkstra.main(): Minimal cost path from {start} to {end}: {' -> '.join(map(str, path))}")
        logging.info(f"dijkstra.main(): Total cost: {cost:.2f}")
    else:
        logging.error(f"dijkstra.main(): No path found from {start} to {end}")

    return path, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the minimal cost path using Dijkstra's algorithm.")
    parser.add_argument("--start", type=int, required=True, help="Start node ID")
    parser.add_argument("--end", type=int, required=True, help="End node ID")
    parser.add_argument("--nodes", type=str, default="q_learning/nodes_12.csv", help="Path to nodes CSV file")
    parser.add_argument("--edges", type=str, default="q_learning/edges_12.csv", help="Path to edges CSV file")
    args = parser.parse_args()

    main(
        args.start,
        args.end,
        args.nodes,
        args.edges
    )
