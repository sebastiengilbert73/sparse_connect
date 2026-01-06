import logging
import argparse
import ast
import random
import numpy as np
import math
import os
import pandas as pd
#from dataclasses import asdict
#import json
import pickle
import sys
sys.path.append("..")
import src.sparse_connect.network as network
from utilities.scheduling import Schedule
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def create_dataset(number_of_nodes, number_of_observations):
    start_target_list = []
    for obs_ndx in range(number_of_observations):
        start_target_list.append((random.randint(0, number_of_nodes - 1), random.randint(0, number_of_nodes - 1)))
    return start_target_list

def validation_stats(net, validation_list):
    trajectory_lengths = []
    total_gains = []
    for start_node, target_node in validation_list:
        visited_nodes, gains = net.hops(start_node, target_node)
        trajectory_lengths.append(len(visited_nodes))
        total_gains.append(math.prod(gains))
    average_length = np.array(trajectory_lengths).mean()
    std_dev_length = np.array(trajectory_lengths).std()
    return average_length, std_dev_length, np.array(total_gains).mean(), np.array(total_gains).std()

def save_heatmap(net, target_node, epoch, output_dir):
    matrix = net.get_adjacency_matrix(target_node)
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
    gainRange,
    maximumHops,
    maximumHopsPenalty,
    trainingSize,
    validationSize,
    schedule,
    visualizationPeriod
):
    logging.info(f"train_network.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the network
    net = network.Network(
        number_of_nodes=numberOfNodes,
        connectivity_average=connectivityAverage,
        connectivity_std_dev=connectivityStdDev,
        gain_range=gainRange,
        maximum_hops=maximumHops,
        maximum_hops_penalty=maximumHopsPenalty
    )

    # Training dataset
    training_list = create_dataset(numberOfNodes, trainingSize)
    # Validation dataset
    validation_list = create_dataset(numberOfNodes, validationSize)

    # Load the schedule
    schedule_df = pd.read_csv(schedule)
    schedule = Schedule(schedule_df)

    viz_dir = os.path.join(outputDirectory, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Select a few random target nodes for visualization
    viz_targets = random.sample(range(numberOfNodes), min(3, numberOfNodes))

    with open(os.path.join(outputDirectory, 'epoch_training.csv'), 'w') as epoch_training_file:
        epoch_training_file.write("epoch,avg_length,std_length,avg_gain,std_gain\n")
        average_length, std_dev_length, average_total_gain, std_dev_total_gain = validation_stats(net, validation_list)
        logging.info(f"train_network.main(): Before training (validation): average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_gain = {average_total_gain}; std_dev_total_gain = {std_dev_total_gain}")
        epoch_training_file.write(
            f"0,{average_length},{std_dev_length},{average_total_gain},{std_dev_total_gain}\n")
        average_lengths = [average_length]
        std_dev_lengths = [std_dev_length]
        average_total_gains = [average_total_gain]
        std_dev_total_gains = [std_dev_total_gain]

        # Initial visualization
        for target in viz_targets:
            save_heatmap(net, target, 0, viz_dir)

        number_of_epochs = schedule.last_epoch()
        for epoch in range(1, number_of_epochs + 1):
            logging.info(f"***** Epoch {epoch} *****")
            for start_node, target_node in training_list:
                visited_nodes, gains = net.hops(start_node, target_node)
                learning_rate = schedule.parameters(epoch)['learning_rate']
                net.update_probabilities(visited_nodes, gains, target_node, learning_rate)

            # Validation
            average_length, std_dev_length, average_total_gain, std_dev_total_gain = validation_stats(net, validation_list)
            logging.info(f"average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_gain = {average_total_gain}; std_dev_total_gain = {std_dev_total_gain}")
            average_lengths.append(average_length)
            std_dev_lengths.append(std_dev_length)
            average_total_gains.append(average_total_gain)
            std_dev_total_gains.append(std_dev_total_gain)
            epoch_training_file.write(f"{epoch},{average_length},{std_dev_length},{average_total_gain},{std_dev_total_gain}\n")
            
            # Save periodic visualizations
            if epoch % visualizationPeriod == 0 or epoch == number_of_epochs:
                for target in viz_targets:
                    save_heatmap(net, target, epoch, viz_dir)

        # Collapse
        net.collapse()
        average_length, std_dev_length, average_total_gain, std_dev_total_gain = validation_stats(net,
                                                                                                                  validation_list)
        logging.info(
            f"After collapse:\naverage_length = {average_length}; std_dev_length = {std_dev_length}; average_total_gain = {average_total_gain}; std_dev_total_gain = {std_dev_total_gain}")
        epoch_training_file.write(
            f"{epoch + 1},{average_length},{std_dev_length},{average_total_gain},{std_dev_total_gain}\n")
        
        # Final visualization after collapse
        for target in viz_targets:
            save_heatmap(net, target, epoch + 1, viz_dir)
    # Save the network
    with open(os.path.join(outputDirectory, 'sparse_network.pkl'), 'wb') as f:
        pickle.dump(net, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train_network'", default='./output_train_network')
    parser.add_argument('--numberOfNodes', help="The number of nodes. Default: 100", type=int, default=100)
    parser.add_argument('--connectivityAverage', help="The approximate connectivity average. Default: 5.0", type=float, default=5.0)
    parser.add_argument('--connectivityStdDev', help="The approximate connectivity standard deviation. Default: 2.0", type=float, default=2.0)
    parser.add_argument('--gainRange', help="The range of linear gain for the links. Default: '[0.01, 1.0]'", default='[0.01, 1.0]')
    parser.add_argument('--maximumHops', help="The maximum number of hops. Default: 100", type=int, default=100)
    parser.add_argument('--maximumHopsPenalty', help="The penalty for reaching the maximum number of hops. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--trainingSize', help="The number of training pairs. Default: 10000", type=int, default=10000)
    parser.add_argument('--validationSize', help="The number of validation pairs. Default: 2000", type=int, default=2000)
    parser.add_argument('--schedule', help="The filepath to the learning schedule. Default: './schedule.csv'", default='./schedule.csv')
    parser.add_argument('--visualizationPeriod', help="The period for epochs where an adjacency matrix is saved. Default: 1", type=int, default=1)
    args = parser.parse_args()
    args.gainRange = ast.literal_eval(args.gainRange)
    main(
        args.outputDirectory,
        args.numberOfNodes,
        args.connectivityAverage,
        args.connectivityStdDev,
        args.gainRange,
        args.maximumHops,
        args.maximumHopsPenalty,
        args.trainingSize,
        args.validationSize,
        args.schedule,
        args.visualizationPeriod
    )