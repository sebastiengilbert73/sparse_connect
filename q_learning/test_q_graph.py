import logging
import argparse
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    graphFilepath,
    startNode,
    targetNode
):
    logging.info(f"test_q_graph.main() graphFilepath = {graphFilepath}")

    with open(graphFilepath, "rb") as f:
        graph = pickle.load(f)

    visited_nodes, costs = graph.trajectory(startNode, targetNode, epsilon=0)
    logging.info(f"trajectory = {visited_nodes}\ncosts = {costs}\ntotal cost = {sum(costs)}")

    return visited_nodes, sum(costs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graphFilepath', help="The filepath to the QGraph, in pickle format")
    parser.add_argument('startNode', help="The start node", type=int)
    parser.add_argument('targetNode', help="The target node", type=int)

    args = parser.parse_args()
    main(
        args.graphFilepath,
        args.startNode,
        args.targetNode
    )