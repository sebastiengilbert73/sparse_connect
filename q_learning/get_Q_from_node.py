import logging
import argparse
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    graphFilepath,
    node
):
    logging.info(f"get_Q_from_node.main()")

    with open(graphFilepath, "rb") as f:
        graph = pickle.load(f)

    Q = graph.QNodes[node].Q
    neighbor_nodes = graph.QNodes[node].neighbor_nodes

    logging.info(f"neighbor_nodes = {neighbor_nodes}\n\nQ =\n{Q}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graphFilepath', help="The filepath to the QGraph, in pickle format")
    parser.add_argument('node', help="The node you want", type=int)

    args = parser.parse_args()
    main(
        args.graphFilepath,
        args.node
    )