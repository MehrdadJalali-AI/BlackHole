import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import logging
from sklearn.preprocessing import MinMaxScaler
import os

logging.basicConfig(level=logging.INFO, filename="bh_evaluation.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_communities(graph):
    """Detects communities using stochastic Louvain algorithm (no seed)."""
    try:
        communities = nx_comm.louvain_communities(graph)  # No seed for stochasticity
        logger.info(f"Detected {len(communities)} communities")
        return communities
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise

def calculate_gravity_per_community(graph, communities, weights=(0.4, 0.4, 0.2)):
    """Calculates normalized gravity per community."""
    community_gravity = {}
    degree_centrality = {}
    betweenness_centrality = {}
    edge_weight_sum = {}
    degree_weight, betweenness_weight, weight_sum_weight = weights

    for community in communities:
        subgraph = graph.subgraph(community)
        if subgraph.number_of_nodes() == 0:
            continue
        degree_centrality_community = nx.degree_centrality(subgraph)
        betweenness_centrality_community = nx.betweenness_centrality(subgraph, normalized=True)
        edge_weight_sum_community = {node: sum(data['weight'] for _, _, data in subgraph.edges(node, data=True)) for node in subgraph.nodes()}

        degree_values = list(degree_centrality_community.values())
        betweenness_values = list(betweenness_centrality_community.values())
        weight_sum_values = list(edge_weight_sum_community.values())

        scaler = MinMaxScaler()
        normalized_degree = scaler.fit_transform(np.array(degree_values).reshape(-1, 1)).flatten()
        normalized_betweenness = scaler.fit_transform(np.array(betweenness_values).reshape(-1, 1)).flatten()
        normalized_weight_sum = scaler.fit_transform(np.array(weight_sum_values).reshape(-1, 1)).flatten()

        for idx, node in enumerate(community):
            degree = normalized_degree[idx]
            betweenness = normalized_betweenness[idx]
            weight_sum = normalized_weight_sum[idx]
            gravity = degree_weight * degree + betweenness_weight * betweenness + weight_sum_weight * weight_sum
            community_gravity[node] = gravity
            degree_centrality[node] = degree_centrality_community[node]
            betweenness_centrality[node] = betweenness_centrality_community[node]
            edge_weight_sum[node] = edge_weight_sum_community[node]

    return community_gravity, degree_centrality, betweenness_centrality, edge_weight_sum

def black_hole_strategy_per_community(graph, gravity, communities, threshold):
    """Removes a percentage of nodes with the lowest gravity in each community."""
    nodes_to_remove = []
    for community in communities:
        community_nodes = [node for node in community if node in gravity]
        if not community_nodes:
            continue
        community_gravity_scores = [(node, gravity[node]) for node in community_nodes]
        community_gravity_scores.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = int((1 - threshold) * len(community_gravity_scores))
        nodes_to_remove.extend([node for node, _ in community_gravity_scores[nodes_to_keep:]])

    graph.remove_nodes_from(nodes_to_remove)
    return graph

def prune_edges(graph, edge_threshold):
    """Prunes edges below a certain weight threshold."""
    edges = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
    if not edges:
        return graph
    edges.sort(key=lambda x: x[2], reverse=True)
    num_edges_to_keep = int((1 - edge_threshold) * len(edges))
    edges_to_keep = edges[:num_edges_to_keep]
    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes())
    new_graph.add_weighted_edges_from(edges_to_keep)
    return new_graph

def save_extended_node_features(graph, original_summary_data, gravity, degree_centrality, betweenness_centrality, edge_weight_sum, communities, filename):
    """Saves extended node features including community IDs."""
    remaining_nodes = list(graph.nodes())
    try:
        original_summary_data = original_summary_data.loc[original_summary_data.index.intersection(remaining_nodes)]
        community_map = {node: idx for idx, community in enumerate(communities) for node in community if node in remaining_nodes}
        metrics_data = pd.DataFrame({
            'refcode': remaining_nodes,
            'Gravity': [gravity.get(node, 0) for node in remaining_nodes],
            'Degree_Centrality': [degree_centrality.get(node, 0) for node in remaining_nodes],
            'Betweenness_Centrality': [betweenness_centrality.get(node, 0) for node in remaining_nodes],
            'Edge_Weight_Sum': [edge_weight_sum.get(node, 0) for node in remaining_nodes],
            'Community_ID': [community_map.get(node, -1) for node in remaining_nodes]
        })
        metrics_data.set_index('refcode', inplace=True)
        final_data = pd.concat([original_summary_data, metrics_data], axis=1)
        final_data.to_csv(filename, index=True)
        logger.info(f"Extended features saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save node features to {filename}: {e}")

def generate_bootstrapped_blackhole_edges(edges_list, n_bootstrap_edges, threshold, run, summary_data_filename):
    """Generates bootstrapped Black Hole sparsified edges."""
    try:
        # Bootstrap sample edges
        sampled_edges = edges_list.sample(n=n_bootstrap_edges, replace=True, random_state=run)
        graph = nx.Graph()
        for _, row in sampled_edges.iterrows():
            graph.add_edge(row['source'], row['target'], weight=row['weight'])
        
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            logger.warning(f"Empty graph for threshold {threshold}, run {run}. Returning empty edges.")
            return pd.DataFrame(columns=['source', 'target', 'weight'])

        communities = detect_communities(graph)
        gravity, degree_centrality, betweenness_centrality, edge_weight_sum = calculate_gravity_per_community(graph, communities)
        graph = black_hole_strategy_per_community(graph, gravity, communities, threshold)
        graph = prune_edges(graph, threshold)
        
        # Save node features
        output_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_blackhole/run_{run}"
        os.makedirs(output_dir, exist_ok=True)
        output_features_filename = os.path.join(output_dir, f"remaining_node_features_t{threshold:.2f}_r{run}.csv")
        save_extended_node_features(graph, pd.read_csv(summary_data_filename, index_col=0), gravity, degree_centrality, betweenness_centrality, edge_weight_sum, communities, output_features_filename)
        
        # Convert to edge list
        edges = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
        bh_edges = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
        bh_edges.to_csv(os.path.join(output_dir, f"BH_edges_t{threshold:.2f}_r{run}.csv"), index=False)
        logger.info(f"Saved BH edges to {output_dir}/BH_edges_t{threshold:.2f}_r{run}.csv")
        return bh_edges
    except Exception as e:
        logger.error(f"Error in BH sparsification for threshold {threshold}, run {run}: {e}")
        return pd.DataFrame(columns=['source', 'target', 'weight'])