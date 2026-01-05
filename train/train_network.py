import logging
import argparse
import ast
import random
import numpy as np
import math
import os
#from dataclasses import asdict
#import json
import pickle
import sys
sys.path.append("..")
import src.sparse_connect.network as network

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(message)s')

def create_dataset(number_of_nodes, number_of_observations):
    start_target_list = []
    for obs_ndx in range(number_of_observations):
        start_target_list.append((random.randint(0, number_of_nodes - 1), random.randint(0, number_of_nodes - 1)))
    return start_target_list

def validation_stats(net, validation_list):
    trajectory_lengths = []
    total_attenuations = []
    for start_node, target_node in validation_list:
        visited_nodes, attenuations = net.hops(start_node, target_node)
        trajectory_lengths.append(len(visited_nodes))
        total_attenuations.append(math.prod(attenuations))
    average_length = np.array(trajectory_lengths).mean()
    std_dev_length = np.array(trajectory_lengths).std()
    return average_length, std_dev_length, np.array(total_attenuations).mean(), np.array(total_attenuations).std()

def main(
    outputDirectory,
    numberOfNodes,
    connectivityAverage,
    connectivityStdDev,
    attenuationRange,
    maximumHops,
    maximumHopsPenalty,
    trainingSize,
    validationSize,
    numberOfEpochs
):
    logging.info(f"train_network.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the network
    net = network.Network(
        number_of_nodes=numberOfNodes,
        connectivity_average=connectivityAverage,
        connectivity_std_dev=connectivityStdDev,
        attenuation_range=attenuationRange,
        maximum_hops=maximumHops,
        maximum_hops_penalty=maximumHopsPenalty
    )

    # Training dataset
    training_list = create_dataset(numberOfNodes, trainingSize)
    # Validation dataset
    validation_list = create_dataset(numberOfNodes, validationSize)

    with open(os.path.join(outputDirectory, 'epoch_training.csv'), 'w') as epoch_training_file:
        epoch_training_file.write("epoch,avg_length,std_length,avg_att,std_att\n")
        average_length, std_dev_length, average_total_attenuation, std_dev_total_attenuation = validation_stats(net, validation_list)
        logging.info(f"train_network.main(): Before training (validation): average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_attenuation = {average_total_attenuation}; std_dev_total_attenuation = {std_dev_total_attenuation}")
        epoch_training_file.write(
            f"0,{average_length},{std_dev_length},{average_total_attenuation},{std_dev_total_attenuation}\n")
        average_lengths = [average_length]
        std_dev_lengths = [std_dev_length]
        average_total_attenuations = [average_total_attenuation]
        std_dev_total_attenuations = [std_dev_total_attenuation]


        for epoch in range(1, numberOfEpochs + 1):
            logging.info(f"***** Epoch {epoch} *****")
            for start_node, target_node in training_list:
                visited_nodes, attenuations = net.hops(start_node, target_node)
                net.update_probabilities(visited_nodes, attenuations, target_node)

            # Validation
            average_length, std_dev_length, average_total_attenuation, std_dev_total_attenuation = validation_stats(net, validation_list)
            logging.info(f"average_length = {average_length}; std_dev_length = {std_dev_length}; average_total_attenuation = {average_total_attenuation}; std_dev_total_attenuation = {std_dev_total_attenuation}")
            average_lengths.append(average_length)
            std_dev_lengths.append(std_dev_length)
            average_total_attenuations.append(average_total_attenuation)
            std_dev_total_attenuations.append(std_dev_total_attenuation)
            epoch_training_file.write(f"{epoch},{average_length},{std_dev_length},{average_total_attenuation},{std_dev_total_attenuation}\n")

        # Collapse
        net.collapse()
        average_length, std_dev_length, average_total_attenuation, std_dev_total_attenuation = validation_stats(net,
                                                                                                                  validation_list)
        logging.info(
            f"After collapse:\naverage_length = {average_length}; std_dev_length = {std_dev_length}; average_total_attenuation = {average_total_attenuation}; std_dev_total_attenuation = {std_dev_total_attenuation}")
        epoch_training_file.write(
            f"{epoch + 1},{average_length},{std_dev_length},{average_total_attenuation},{std_dev_total_attenuation}\n")
    # Save the network
    with open(os.path.join(outputDirectory, 'sparse_network.pkl'), 'wb') as f:
        pickle.dump(net, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train_network'", default='./output_train_network')
    parser.add_argument('--numberOfNodes', help="The number of nodes. Default: 100", type=int, default=100)
    parser.add_argument('--connectivityAverage', help="The approximate connectivity average. Default: 5.0", type=float, default=5.0)
    parser.add_argument('--connectivityStdDev', help="The approximate connectivity standard deviation. Default: 2.0", type=float, default=2.0)
    parser.add_argument('--attenuationRange', help="The range of attenuation for the links. Default: '[0.8, 1.0]'", default='[0.8, 1.0]')
    parser.add_argument('--maximumHops', help="The maximum number of hops. Default: 100", type=int, default=100)
    parser.add_argument('--maximumHopsPenalty', help="The penalty for reaching the maximum number of hops. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--trainingSize', help="The number of training pairs. Default: 10000", type=int, default=10000)
    parser.add_argument('--validationSize', help="The number of validation pairs. Default: 2000", type=int, default=2000)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 1000", type=int, default=1000)
    args = parser.parse_args()
    args.attenuationRange = ast.literal_eval(args.attenuationRange)
    main(
        args.outputDirectory,
        args.numberOfNodes,
        args.connectivityAverage,
        args.connectivityStdDev,
        args.attenuationRange,
        args.maximumHops,
        args.maximumHopsPenalty,
        args.trainingSize,
        args.validationSize,
        args.numberOfEpochs
    )