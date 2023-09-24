import numpy as np
import os
import torch

import ipdb

def tsne_plot_data(x, label, epoch, iteration, train=False):
    # save x and label in npy format
    if train:
        if not os.path.exists("./train_tsne"):
            os.makedirs("./train_tsne")
        np.save(f"./train_tsne/x_epoch_{epoch}_iter_{iteration}.npy", x.cpu().detach().numpy())
        np.save(f"./train_tsne/label_epoch_{epoch}_iter_{iteration}.npy", label.cpu().detach().numpy())
    else:
        if not os.path.exists("./test_tsne"):
            os.makedirs("./test_tsne")
        np.save(f"./test_tsne/x_epoch_{epoch}_iter_{iteration}.npy", x.cpu().detach().numpy())
        np.save(f"./test_tsne/label_epoch_{epoch}_iter_{iteration}.npy", label.cpu().detach().numpy())


def node_att_data_save(A, epoch, iteration, labels, learned_he_weights, hyperedge_index, batch, train=False):
    # Save attention and labels as before
    if train:
        if not os.path.exists("./train_node_att"):
            os.makedirs("./train_node_att")
        np.save(
            f"./train_node_att/A_epoch_{epoch}_iter_{iteration}.npy", A.cpu().detach().numpy())
        np.save(
            f"./train_node_att/label_epoch_{epoch}_iter_{iteration}.npy", labels.cpu().detach().numpy())


        # Check if learned weights file doesn't exist
        learned_weights_path = f"./train_node_att/learned_weights_epoch_{epoch}.npy"
        if not os.path.exists(learned_weights_path):
            np.save(learned_weights_path,
                    learned_he_weights.cpu().detach().numpy())

        # Check if hyperedge indices combined file doesn't exist
        hyperedge_indices_combined_path = f"./train_node_att/hyperedge_indices_combined.npy"
        if not os.path.exists(hyperedge_indices_combined_path):
            unique_batches = batch.unique().cpu().numpy()
            batch_size = len(unique_batches)

            combined_matrices = torch.zeros(batch_size, 400, 400)

            for idx, graph_idx in enumerate(unique_batches):
                graph_indices = (batch == graph_idx).nonzero(as_tuple=True)[0]
                min_index = graph_indices.min()

                mask = (hyperedge_index[0] >= min_index) & (
                    hyperedge_index[0] < min_index + 400)
                graph_hyperedge = hyperedge_index[:, mask]

                # Remove the offset
                graph_hyperedge[0] = graph_hyperedge[0] - min_index
                graph_hyperedge[1] = graph_hyperedge[1] - min_index

                # Convert to [400, 400] format for the current graph and assign to the combined tensor
                matrix_representation = torch.zeros(400, 400)
                # Assuming binary hyperedges
                matrix_representation[graph_hyperedge[0],
                                      graph_hyperedge[1]] = 1
                combined_matrices[idx] = matrix_representation

            np.save(hyperedge_indices_combined_path,
                    combined_matrices.cpu().numpy())
