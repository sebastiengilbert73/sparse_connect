import logging
import argparse
import ast
import os
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    outputDirectory,
    numberOfNodes,
    connectivityAverage,
    connectivityStdDev,
    costRange,
    costNumberOfDigits,
    allowSelfLinks
):
    logging.info("generate_nodes_edges.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    with open(os.path.join(outputDirectory, "nodes.csv"), 'w') as nodes_file:
        nodes_file.write("Id,Label\n")
        for node_ndx in range(numberOfNodes):
            nodes_file.write(f"{node_ndx},Node-{node_ndx}\n")

    with open(os.path.join(outputDirectory, "edges.csv"), 'w') as edges_file:
        edges_file.write("Source,Target,Type,Weight\n")
        for node_ndx in range(numberOfNodes):
            number_of_neighbors = connectivityAverage + connectivityStdDev * random.random()
            number_of_neighbors = round(number_of_neighbors)
            number_of_neighbors = max(number_of_neighbors, 2)  # At least two out-connections
            number_of_neighbors = min(number_of_neighbors, numberOfNodes if allowSelfLinks else numberOfNodes - 1)  # Not more than N connections
            candidate_nodes = list(range(numberOfNodes))
            if not allowSelfLinks:
                candidate_nodes = [i for i in range(numberOfNodes) if i != node_ndx]
            neighbors = random.sample(candidate_nodes, number_of_neighbors)
            for neighbor in neighbors:
                cost = costRange[0] + (costRange[1] - costRange[0]) * random.random()
                cost = round(cost, costNumberOfDigits)
                edges_file.write(f"{node_ndx},{neighbor},Directed,{cost}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_generate_nodes_edges'",
                        default='./output_generate_nodes_edges')
    parser.add_argument('--numberOfNodes', help="The number of nodes. Default: 8", type=int, default=8)
    parser.add_argument('--connectivityAverage', help="The approximate connectivity average. Default: 2.0", type=float,
                        default=2.0)
    parser.add_argument('--connectivityStdDev', help="The approximate connectivity standard deviation. Default: 0.0",
                        type=float, default=0.0)
    parser.add_argument('--costRange', help="The range of cost for the links. Default: '[0.05, 0.95]'",
                        default='[0.05, 0.95]')
    parser.add_argument('--costNumberOfDigits', help="Round the cost to this number of digits. Default: 1", type=int, default=1)
    parser.add_argument('--allowSelfLinks', help="Allow links that point towards their own starting node", action='store_true')

    args = parser.parse_args()
    args.costRange = ast.literal_eval(args.costRange)
    main(
        args.outputDirectory,
        args.numberOfNodes,
        args.connectivityAverage,
        args.connectivityStdDev,
        args.costRange,
        args.costNumberOfDigits,
        args.allowSelfLinks
    )