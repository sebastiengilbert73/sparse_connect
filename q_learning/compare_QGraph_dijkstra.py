import logging
import argparse
import pickle
import os
import test_q_graph
import dijkstra

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    outputDirectory,
    graphFilepath,
    nodesFilepath,
    edgesFilepath
):
    logging.info("compare_QGraph_dijkstra.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    with open(graphFilepath, "rb") as f:
        graph = pickle.load(f)

    number_of_nodes = graph.number_of_nodes
    start_target_list = []
    for start_node in range(number_of_nodes):
        for target_node in range(number_of_nodes):
            if target_node != start_node:
                start_target_list.append((start_node, target_node))

    with open(os.path.join(outputDirectory, "comparison.csv"), 'w') as output_file:
        output_file.write("trajectory_QGraph,trajectory_dijkstra,cost_QGraph,cost_dijkstra\n")
        for start_node, target_node in start_target_list:
            trajectory_qgraph, costs_qgraph = graph.trajectory(start_node, target_node, epsilon=0)
            trajectory_dijkstra, cost_dijkstra = dijkstra.main(
                start=start_node,
                end=target_node,
                nodes=nodesFilepath,
                edges=edgesFilepath
            )
            if trajectory_dijkstra is None:
                trajectory_dijkstra = 'None'
            output_file.write(f"[{' '.join(map(str, trajectory_qgraph))}],[{' '.join(map(str, trajectory_dijkstra))}],{sum(costs_qgraph)},{cost_dijkstra}\n")
            if trajectory_qgraph != trajectory_dijkstra:
                logging.info(f"compare_QGraph_dijkstra.main(): start_node = {start_node}; target_node = {target_node}\ntrajectory_qgraph = {trajectory_qgraph}\ntrajectory_dijkstra = {trajectory_dijkstra}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_compare_QGraph_dijkstra'", default='./output_compare_QGraph_dijkstra')
    parser.add_argument('graphFilepath', help="The filepath to the QGraph, in pickle format")
    parser.add_argument('nodesFilepath', help="The filepath to the nodes csv file")
    parser.add_argument('edgesFilepath', help="The filepath to the edges csv file")

    args = parser.parse_args()

    main(
        args.outputDirectory,
        args.graphFilepath,
        args.nodesFilepath,
        args.edgesFilepath
    )