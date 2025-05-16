import pandas as pd
import numpy as np
import networkx as nx
import logging
import os

logging.basicConfig(level=logging.INFO, filename="bh_evaluation.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def sparsify_edges(edges_list, target_num_nodes, method, valid_nodes, target_num_edges):
    """Sparsifies edges to match target number of nodes and edges."""
    if target_num_nodes == 0 or target_num_edges == 0:
        logger.warning(f"No nodes or edges to keep for method {method}")
        return pd.DataFrame(columns=edges_list.columns)

    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(valid_nodes)
    selected_edges = []
    selected_nodes = set()

    if method == "blackhole":
        return edges_list

    elif method == "random":
        shuffled_edges = edges_list.sample(frac=1, random_state=42).reset_index(drop=True)
        for _, edge in shuffled_edges.iterrows():
            src, tgt = edge['source'], edge['target']
            if src in valid_nodes and tgt in valid_nodes:
                temp_graph.add_edge(src, tgt, weight=edge['weight'])
                selected_edges.append(edge)
                selected_nodes.update([src, tgt])
                if len(selected_edges) >= target_num_edges and len(selected_nodes) >= target_num_nodes:
                    break
        # If edge count is still below target, add more edges between selected nodes
        if len(selected_edges) < target_num_edges and selected_nodes:
            remaining_edges = edges_list[
                (edges_list['source'].isin(selected_nodes)) & (edges_list['target'].isin(selected_nodes)) &
                (~edges_list.index.isin([e.name for e in selected_edges]))
            ].sample(frac=1, random_state=42).reset_index(drop=True)
            for _, edge in remaining_edges.iterrows():
                src, tgt = edge['source'], edge['target']
                temp_graph.add_edge(src, tgt, weight=edge['weight'])
                selected_edges.append(edge)
                if len(selected_edges) >= target_num_edges:
                    break
        if len(selected_edges) < target_num_edges:
            logger.warning(f"Random method could only select {len(selected_edges)} edges (target: {target_num_edges})")
        return pd.DataFrame(selected_edges, columns=edges_list.columns)

    elif method == "stratified":
        bins = np.histogram(edges_list['weight'], bins=10)[1]
        for i in range(len(bins) - 1):
            bin_mask = (edges_list['weight'] >= bins[i]) & (edges_list['weight'] < bins[i + 1])
            bin_edges = edges_list[bin_mask].sample(frac=1, random_state=42).reset_index(drop=True)
            for _, edge in bin_edges.iterrows():
                src, tgt = edge['source'], edge['target']
                if src in valid_nodes and tgt in valid_nodes:
                    temp_graph.add_edge(src, tgt, weight=edge['weight'])
                    selected_edges.append(edge)
                    selected_nodes.update([src, tgt])
                    if len(selected_edges) >= target_num_edges and len(selected_nodes) >= target_num_nodes:
                        break
            if len(selected_edges) >= target_num_edges and len(selected_nodes) >= target_num_nodes:
                break
        # If edge count is still below target, add more edges between selected nodes
        if len(selected_edges) < target_num_edges and selected_nodes:
            remaining_edges = edges_list[
                (edges_list['source'].isin(selected_nodes)) & (edges_list['target'].isin(selected_nodes)) &
                (~edges_list.index.isin([e.name for e in selected_edges]))
            ].sample(frac=1, random_state=42).reset_index(drop=True)
            for _, edge in remaining_edges.iterrows():
                src, tgt = edge['source'], edge['target']
                temp_graph.add_edge(src, tgt, weight=edge['weight'])
                selected_edges.append(edge)
                if len(selected_edges) >= target_num_edges:
                    break
        if len(selected_edges) < target_num_edges:
            logger.warning(f"Stratified method could only select {len(selected_edges)} edges (target: {target_num_edges})")
        return pd.DataFrame(selected_edges, columns=edges_list.columns)

    else:
        raise ValueError(f"Unknown sparsification method: {method}")

def save_sparsified_edges(sparse_edges, threshold, method, run):
    """Saves sparsified edges for non-BH methods."""
    output_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
    os.makedirs(output_dir, exist_ok=True)
    sparse_edges.to_csv(os.path.join(output_dir, f"edges_t{threshold:.2f}_r{run}.csv"), index=False)
    logger.info(f"Saved {method} edges to {output_dir}/edges_t{threshold:.2f}_r{run}.csv")