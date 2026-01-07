import logging
import argparse
import ast
import os
import q_graph
import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import sys
sys.path.append("..")
from utilities.scheduling import Schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def create_dataset(number_of_nodes, number_of_observations):
    start_target_list = []
    for obs_ndx in range(number_of_observations):
        start_target_list.append((random.randint(0, number_of_nodes - 1), random.randint(0, number_of_nodes - 1)))
    return start_target_list

def validation_stats(graph, validation_list, epsilon=0):
    trajectory_lengths = []
    total_costs = []
    for start_node, target_node in validation_list:
        visited_nodes, costs = graph.trajectory(start_node, target_node, epsilon)
        trajectory_lengths.append(len(visited_nodes))
        total_costs.append(sum(costs))
    average_length = np.array(trajectory_lengths).mean()
    std_dev_length = np.array(trajectory_lengths).std()
    return average_length, std_dev_length, np.array(total_costs).mean(), np.array(total_costs).std()

def save_heatmap(graph, target_node, epoch, output_dir):
    matrix = graph.get_adjacency_matrix(target_node)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", cbar=True)
    plt.title(f"Connection Probabilities - Epoch {epoch} - Target {target_node}")
    plt.xlabel("Destination Node")
    plt.ylabel("Origin Node")
    filepath = os.path.join(output_dir, f"heatmap_epoch_{epoch}_target_{target_node}.png")
    plt.savefig(filepath)
    plt.close()

def main(
    outputDirectory,
    numberOfNodes,
    connectivityAverage,
    connectivityStdDev,
    costRange,
    maximumHops,
    maximumHopsPenalty,
    trainingSize,
    validationSize,
    schedule,
    visualizationPeriod
):
    logging.info("q_learn.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Validation dataset
    validation_list = create_dataset(numberOfNodes, validationSize)

    # Load the schedule
    schedule_df = pd.read_csv(schedule)
    schedule = Schedule(schedule_df)

    # Create the graph
    graph = q_graph.QGraph(
        number_of_nodes=numberOfNodes,
        connectivity_average=connectivityAverage,
        connectivity_std_dev=connectivityStdDev,
        cost_range=costRange,
        maximum_hops=maximumHops,
        maximum_hops_penalty=maximumHopsPenalty
    )

    viz_dir = os.path.join(outputDirectory, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    with open(os.path.join(outputDirectory, 'epoch_learning.csv'), 'w') as epoch_learning_file:
        epoch_learning_file.write("epoch,avg_length,std_length,avg_cost,std_cost\n")
        average_length, std_dev_length, average_total_cost, std_dev_total_cost = validation_stats(graph, validation_list, epsilon=0)
        logging.info(f"train_network.main(): Before training (validation): average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_cost = {average_total_cost}; std_dev_total_cost = {std_dev_total_cost}")
        epoch_learning_file.write(
            f"0,{average_length},{std_dev_length},{average_total_cost},{std_dev_total_cost}\n")
        average_lengths = [average_length]
        std_dev_lengths = [std_dev_length]
        average_total_costs = [average_total_cost]
        std_dev_total_costs = [std_dev_total_cost]

        # Initial visualization
        save_heatmap(graph, target_node=numberOfNodes - 1, epoch=0, output_dir=outputDirectory)

        number_of_epochs = schedule.last_epoch()
        for epoch in range(1, number_of_epochs + 1):
            logging.info(f"***** Epoch {epoch} *****")
            epsilon = schedule.parameters(epoch)['epsilon']
            alpha = schedule.parameters(epoch)['alpha']
            gamma = schedule.parameters(epoch)['gamma']

            # Training dataset
            training_list = create_dataset(numberOfNodes, trainingSize)

            for start_node, target_node in training_list:
                visited_nodes, costs = graph.trajectory(start_node, target_node, epsilon)
                graph.update_Q(visited_nodes, costs, alpha, gamma, target_node)

            # Validation
            average_length, std_dev_length, average_total_cost, std_dev_total_cost = \
                validation_stats(graph, validation_list, epsilon=epsilon)
            logging.info(
                f"average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_cost = {average_total_cost}; std_dev_total_cost = {std_dev_total_cost}")
            average_lengths.append(average_length)
            std_dev_lengths.append(std_dev_length)
            average_total_costs.append(average_total_cost)
            std_dev_total_costs.append(std_dev_total_cost)
            epoch_learning_file.write(
                f"{epoch},{average_length},{std_dev_length},{average_total_cost},{std_dev_total_cost}\n")

        # With epsilon=0
        average_length, std_dev_length, average_total_cost, std_dev_total_cost = \
            validation_stats(graph, validation_list, epsilon=0)
        logging.info(
            f"average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_cost = {average_total_cost}; std_dev_total_cost = {std_dev_total_cost}")
        epoch_learning_file.write(
            f"{epoch + 1},{average_length},{std_dev_length},{average_total_cost},{std_dev_total_cost}\n")

    # Save the network
    with open(os.path.join(outputDirectory, 'QGraph.pkl'), 'wb') as f:
        pickle.dump(graph, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_q_learn'",
                        default='./output_q_learn')
    parser.add_argument('--numberOfNodes', help="The number of nodes. Default: 100", type=int, default=100)
    parser.add_argument('--connectivityAverage', help="The approximate connectivity average. Default: 5.0", type=float,
                        default=5.0)
    parser.add_argument('--connectivityStdDev', help="The approximate connectivity standard deviation. Default: 2.0",
                        type=float, default=2.0)
    parser.add_argument('--costRange', help="The range of cost for the links. Default: '[0.0, 1.0]'",
                        default='[0.0, 1.0]')
    parser.add_argument('--maximumHops', help="The maximum number of hops. Default: 100", type=int, default=100)
    parser.add_argument('--maximumHopsPenalty',
                        help="The penalty for reaching the maximum number of hops. Default: 1.0", type=float,
                        default=1.0)
    parser.add_argument('--trainingSize', help="The number of training pairs. Default: 10000", type=int, default=10000)
    parser.add_argument('--validationSize', help="The number of validation pairs. Default: 2000", type=int,
                        default=2000)
    parser.add_argument('--schedule', help="The filepath to the learning schedule. Default: './schedule.csv'",
                        default='./schedule.csv')
    parser.add_argument('--visualizationPeriod',
                        help="The period for epochs where an adjacency matrix is saved. Default: 1", type=int,
                        default=1)
    args = parser.parse_args()
    args.costRange = ast.literal_eval(args.costRange)
    main(
        args.outputDirectory,
        args.numberOfNodes,
        args.connectivityAverage,
        args.connectivityStdDev,
        args.costRange,
        args.maximumHops,
        args.maximumHopsPenalty,
        args.trainingSize,
        args.validationSize,
        args.schedule,
        args.visualizationPeriod
    )