import time
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import modularity
import torch
from sklearn.model_selection import train_test_split
import logging
import os
import psutil
from tqdm import tqdm
from data_utils import load_edges_list, load_summary_data
from bh_sparsification import generate_bootstrapped_blackhole_edges
from sparsification_methods import sparsify_edges, save_sparsified_edges
from graphsage_model import GraphSAGE, GCN, GAT, train, test
from experiment_manager import load_checkpoint, save_checkpoint, save_results, aggregate_results

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("bh_evaluation.log", mode="a")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
handler.buffering = 1
logger.addHandler(handler)
logger.propagate = False

def log_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024**2  # MB
    logger.info(f"Memory usage: {mem:.2f} MB")

if __name__ == "__main__":
    start_time = time.time()
    thresholds = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.06, 0.03, 0]
    methods = ["blackhole", "random", "stratified"]
    models = ["GraphSAGE", "GCN", "GAT"]
    num_runs = 10
    edges_list_filename = 'MOFGalaxyNet.csv'
    summary_data_filename = 'MOFCSD.csv'
    checkpoint_file = 'bh_evaluation_checkpoint.json'
    debug_mode = False
    use_edge_weights = True

    try:
        logger.info("Starting Bootstrapped Black Hole Strategy and Evaluation")
        log_memory_usage()

        # Load data
        try:
            edges_list = load_edges_list(edges_list_filename)
            if not all(col in edges_list.columns for col in ['source', 'target', 'weight']):
                raise ValueError("MOFGalaxyNet.csv must contain 'source', 'target', 'weight' columns")
            logger.info(f"Loaded MOFGalaxyNet.csv: {len(edges_list)} edges")
            node_labels = pd.concat([edges_list['source'], edges_list['target']]).unique()
            features_df, summary_data = load_summary_data(summary_data_filename, node_labels)
            logger.info(f"Loaded MOFCSD.csv: {len(summary_data)} nodes")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            exit(1)

        # Validate node consistency
        try:
            summary_nodes = set(summary_data.index)
            edge_nodes = set(node_labels)
            missing_nodes = edge_nodes - summary_nodes
            if missing_nodes:
                logger.warning(f"{len(missing_nodes)} nodes in MOFGalaxyNet.csv not in MOFCSD.csv: {list(missing_nodes)[:10]}")
        except Exception as e:
            logger.error(f"Failed to validate nodes: {e}")
            exit(1)

        # Construct initial graph
        graph = nx.Graph()
        try:
            for _, row in edges_list.iterrows():
                graph.add_edge(row['source'], row['target'], weight=row['weight'])
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                raise ValueError(f"Initial graph is empty: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            initial_num_nodes = graph.number_of_nodes()
            initial_num_edges = graph.number_of_edges()
            logger.info(f"Initial graph: {initial_num_nodes} nodes, {initial_num_edges} edges")
        except Exception as e:
            logger.error(f"Failed to construct initial graph: {e}")
            exit(1)

        # Prepare features and labels
        try:
            required_columns = ['Pore Limiting Diameter', 'linker SMILES', 'metal', 'Largest Cavity Diameter', 'Largest Free Sphere']
            if not all(col in summary_data.columns for col in required_columns):
                raise ValueError(f"MOFCSD.csv missing required columns: {required_columns}")
            summary_data['PLD_category'] = pd.cut(
                summary_data['Pore Limiting Diameter'],
                bins=[-np.inf, 2.4, 4.4, 5.9, np.inf],
                labels=['nonporous', 'small pore', 'medium pore', 'large pore']
            )
            labels = summary_data['PLD_category'].astype('category').cat.codes.to_numpy()
            x = torch.tensor(features_df.values, dtype=torch.float)
            y = torch.tensor(labels, dtype=torch.long)
            logger.info(f"Prepared features with shape {x.shape} and labels with shape {y.shape}")
            label_counts = pd.Series(labels).value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
            class_counts = np.bincount(labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = torch.tensor(class_weights, dtype=torch.float) / class_weights.sum()
            logger.info(f"Class weights: {class_weights.tolist()}")
        except Exception as e:
            logger.error(f"Failed to prepare features and labels: {e}")
            exit(1)

        # Create fixed global test set
        train_indices, test_indices = train_test_split(np.arange(len(summary_data)), test_size=0.2, random_state=42, stratify=labels)
        test_mask = torch.tensor(test_indices, dtype=torch.long)
        valid_nodes = set(summary_data.iloc[train_indices].index)
        logger.info(f"Train set: {len(train_indices)} nodes, Test set: {len(test_indices)} nodes")

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_file)
        completed_sparsification = set(tuple(x) for x in checkpoint.get("completed_sparsification", []))
        completed_evaluation = set(tuple(x) for x in checkpoint.get("completed_evaluation", []))
        results = checkpoint.get("results", [])
        logger.info(f"Loaded checkpoint: {len(completed_sparsification)} sparsifications, {len(completed_evaluation)} evaluations completed")

        # Debug mode
        if debug_mode:
            thresholds = [0.95]
            num_runs = 1
            runs = [6]
        else:
            runs = range(num_runs)

        # Total experiments
        total_experiments = len(thresholds) * len(runs) * len(methods) * len(models)
        experiment_counter = 0

        # Main Loop
        with tqdm(total=total_experiments, desc="Overall Progress", position=0) as pbar:
            for threshold in thresholds:
                logger.info(f"Processing threshold: {threshold:.2f}")
                for run in runs:
                    log_memory_usage()
                    # Black Hole Sparsification
                    sparsification_id = (threshold, run)
                    bh_edge_file = f"sparsified_graphs/threshold_{threshold:.2f}/method_blackhole/run_{run}/BH_edges_t{threshold:.2f}_r{run}.csv"
                    if sparsification_id in completed_sparsification and os.path.exists(bh_edge_file):
                        logger.info(f"Skipping completed BH sparsification: threshold={threshold}, run={run}")
                        try:
                            bh_edges = pd.read_csv(bh_edge_file)
                            logger.info(f"Loaded BH edges: {len(bh_edges)} edges")
                        except Exception as e:
                            logger.error(f"Failed to load BH edges from {bh_edge_file}: {e}")
                            pbar.update(len(methods) * len(models))
                            continue
                    else:
                        logger.info(f"Generating BH edges for threshold {threshold}, run={run}")
                        try:
                            n_bootstrap_edges = len(edges_list)
                            bh_edges = generate_bootstrapped_blackhole_edges(edges_list, n_bootstrap_edges, threshold, run, summary_data_filename)
                            if bh_edges.empty:
                                logger.warning(f"Empty BH edges for threshold {threshold}, run={run}. Skipping.")
                                pbar.update(len(methods) * len(models))
                                continue
                            os.makedirs(os.path.dirname(bh_edge_file), exist_ok=True)
                            bh_edges.to_csv(bh_edge_file, index=False)
                            completed_sparsification.add(sparsification_id)
                            checkpoint = {
                                "completed_sparsification": list(completed_sparsification),
                                "completed_evaluation": list(completed_evaluation),
                                "results": results
                            }
                            save_checkpoint(checkpoint_file, checkpoint)
                            logger.info(f"Saved BH edges: {len(bh_edges)} edges")
                        except Exception as e:
                            logger.error(f"Failed to generate BH edges for threshold {threshold}, run={run}: {e}")
                            pbar.update(len(methods) * len(models))
                            continue

                    num_edges_bh = len(bh_edges)
                    target_num_nodes = len(set(nx.from_pandas_edgelist(bh_edges, 'source', 'target', 'weight').nodes()))
                    if target_num_nodes == 0:
                        logger.warning(f"No connected nodes for threshold {threshold}, run={run}. Skipping.")
                        pbar.update(len(methods) * len(models))
                        continue
                    logger.info(f"BH graph: {target_num_nodes} nodes, {num_edges_bh} edges")

                    for method in methods:
                        edges_list_method = bh_edges.copy() if method == "blackhole" else edges_list
                        logger.info(f"Performing sparsification for method {method} at threshold {threshold}, run={run}")
                        try:
                            sparse_edges = sparsify_edges(edges_list_method, target_num_nodes, method, valid_nodes=valid_nodes, target_num_edges=num_edges_bh)
                            logger.info(f"Method: {method}, Input edges: {len(edges_list_method)}, Output edges: {len(sparse_edges)}")
                        except Exception as e:
                            logger.error(f"Failed to sparsify edges for method {method}, threshold {threshold}, run {run}: {e}")
                            pbar.update(len(models))
                            continue

                        if method != "blackhole":
                            try:
                                save_sparsified_edges(sparse_edges, threshold, method, run)
                                logger.info(f"Saved sparsified edges for method {method}")
                            except Exception as e:
                                logger.error(f"Failed to save sparsified edges for method {method}, threshold {threshold}, run {run}: {e}")
                                pbar.update(len(models))
                                continue

                        graph = nx.from_pandas_edgelist(sparse_edges, 'source', 'target', 'weight')
                        if graph.number_of_edges() == 0:
                            logger.warning(f"Empty graph for method {method}, threshold {threshold}, run {run}. Skipping.")
                            pbar.update(len(models))
                            continue

                        num_edges = graph.number_of_edges()
                        num_nodes = len(set(nx.subgraph(graph, [n for n in graph.nodes() if graph.degree(n) > 0]).nodes()))
                        logger.info(f"Graph has {num_nodes} nodes and {num_edges} edges (target: {target_num_nodes} nodes, {num_edges_bh} edges)")

                        # Validate graph nodes
                        graph_nodes = set(graph.nodes())
                        common_nodes = graph_nodes & summary_nodes
                        logger.info(f"Graph nodes: {len(graph_nodes)}, Common with summary_data: {len(common_nodes)}")

                        # Compute modularity and additional metrics
                        modularity_score = 0.0
                        num_communities = 0
                        avg_community_size = 0.0
                        avg_clustering = 0.0
                        graph_density = 0.0
                        avg_degree = 0.0
                        communities = []
                        try:
                            if method == "blackhole":
                                node_features_file = f"sparsified_graphs/threshold_{threshold:.2f}/method_blackhole/run_{run}/remaining_node_features_t{threshold:.2f}_r{run}.csv"
                                if not os.path.exists(node_features_file):
                                    logger.error(f"Node features file not found: {node_features_file}")
                                else:
                                    node_features = pd.read_csv(node_features_file)
                                    if 'Community_ID' not in node_features.columns:
                                        logger.error(f"Community_ID column missing in {node_features_file}")
                                    else:
                                        community_ids = node_features['Community_ID'].values
                                        nodes = node_features['refcode'].values
                                        max_community_id = int(community_ids.max()) + 1
                                        communities = [set() for _ in range(max_community_id)]
                                        for node, cid in zip(nodes, community_ids):
                                            if cid != -1:
                                                communities[int(cid)].add(node)
                                        communities = [c for c in communities if c]
                                        partition_nodes = set().union(*communities)
                                        if partition_nodes != graph_nodes or any(len(c) == 0 for c in communities):
                                            logger.warning(f"Invalid BH partition for method {method}, threshold {threshold}, run {run}. Falling back to Louvain.")
                                            communities = nx_comm.louvain_communities(graph)
                                        modularity_score = modularity(graph, communities, weight='weight')
                            else:
                                communities = nx_comm.louvain_communities(graph)
                                modularity_score = modularity(graph, communities, weight='weight')
                            
                            # Calculate additional metrics
                            num_communities = len(communities)
                            avg_community_size = sum(len(c) for c in communities) / num_communities if num_communities > 0 else 0.0
                            avg_clustering = nx.average_clustering(graph, weight='weight')
                            graph_density = nx.density(graph) if num_nodes > 1 else 0.0
                            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
                            
                            logger.info(f"Metrics for method {method}, threshold {threshold}, run {run}: Modularity={modularity_score:.4f}, "
                                       f"Num_Communities={num_communities}, Avg_Community_Size={avg_community_size:.2f}, "
                                       f"Avg_Clustering={avg_clustering:.4f}, Graph_Density={graph_density:.4f}, "
                                       f"Avg_Degree={avg_degree:.2f}")
                        except Exception as e:
                            logger.error(f"Failed to compute metrics for method {method}, threshold {threshold}, run {run}: {e}")
                            communities = nx_comm.louvain_communities(graph)
                            modularity_score = modularity(graph, communities, weight='weight')
                            num_communities = len(communities)
                            avg_community_size = sum(len(c) for c in communities) / num_communities if num_communities > 0 else 0.0
                            avg_clustering = nx.average_clustering(graph, weight='weight')
                            graph_density = nx.density(graph) if num_nodes > 1 else 0.0
                            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
                            logger.info(f"Fallback metrics for method {method}, threshold {threshold}, run {run}: Modularity={modularity_score:.4f}, "
                                       f"Num_Communities={num_communities}, Avg_Community_Size={avg_community_size:.2f}, "
                                       f"Avg_Clustering={avg_clustering:.4f}, Graph_Density={graph_density:.4f}, "
                                       f"Avg_Degree={avg_degree:.2f}")

                        # Prepare GraphSAGE data
                        node_to_index = {node: idx for idx, node in enumerate(graph_nodes)}
                        logger.info(f"Node to index mapping: {len(node_to_index)} nodes")
                        try:
                            weights = np.array([graph[src][dst]['weight'] for src, dst in graph.edges if src in node_to_index and dst in node_to_index])
                            logger.info(f"Valid edges for GNN: {weights.size}, Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
                            if weights.size == 0:
                                logger.warning(f"No valid edges for method {method}, threshold {threshold}, run {run}. Skipping.")
                                pbar.update(len(models))
                                continue
                            if use_edge_weights and (weights.min() < 0 or weights.max() > 1):
                                logger.info(f"Normalizing edge weights for method {method}, threshold {threshold}, run {run}: min={weights.min():.4f}, max={weights.max():.4f}")
                                weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
                            edge_weight = torch.tensor(weights, dtype=torch.float) if use_edge_weights else None
                        except Exception as e:
                            logger.error(f"Failed to process edge weights for method {method}, threshold {threshold}, run {run}: {e}")
                            pbar.update(len(models))
                            continue

                        edge_index = torch.tensor(
                            [(node_to_index[src], node_to_index[dst]) for src, dst in graph.edges if src in node_to_index and dst in node_to_index],
                            dtype=torch.long
                        ).t().contiguous()

                        # Subset features and labels for graph nodes
                        graph_indices = [summary_data.index.get_loc(node) for node in graph_nodes]
                        x_subset = x[graph_indices]
                        y_subset = y[graph_indices]
                        train_mask_subset = torch.tensor([i for i, idx in enumerate(graph_indices) if idx in train_indices], dtype=torch.long)
                        test_mask_subset = torch.tensor([i for i, idx in enumerate(graph_indices) if idx in test_indices], dtype=torch.long)
                        logger.info(f"GNN data: {len(graph_indices)} nodes, train_mask: {len(train_mask_subset)}, test_mask: {len(test_mask_subset)}")
                        logger.info(f"Label distribution in graph: {dict(pd.Series(y_subset.numpy()).value_counts())}")

                        # Fallback if test_mask is empty
                        if test_mask_subset.size(0) == 0:
                            logger.warning(f"Empty test mask for method {method}, threshold {threshold}, run {run}, using 20% of graph nodes for testing")
                            train_idx, test_idx = train_test_split(np.arange(len(graph_indices)), test_size=0.2, random_state=42, stratify=y_subset.numpy())
                            train_mask_subset = torch.tensor(train_idx, dtype=torch.long)
                            test_mask_subset = torch.tensor(test_idx, dtype=torch.long)
                            logger.info(f"Fallback masks: train_mask: {len(train_mask_subset)}, test_mask: {len(test_mask_subset)}")

                        # Create data object
                        data = type('Data', (), {})()
                        data.x = x_subset
                        data.edge_index = edge_index
                        data.edge_weight = edge_weight
                        data.y = y_subset
                        data.train_mask = train_mask_subset
                        data.test_mask = test_mask_subset

                        for model_name in models:
                            experiment_counter += 1
                            experiment_id = (threshold, method, run, model_name)
                            progress = f"Experiment {experiment_counter}/{total_experiments} (threshold={threshold:.2f}, method={method}, run={run}, model={model_name})"
                            logger.info(f"Starting {progress}")

                            if experiment_id in completed_evaluation:
                                logger.info(f"Skipping completed experiment: {progress}")
                                pbar.update(1)
                                continue

                            # Create evaluation folder
                            eval_dir = f"evaluation/threshold_{threshold:.2f}/method_{method}/run_{run}/model_{model_name}"
                            os.makedirs(eval_dir, exist_ok=True)

                            # Train and evaluate model
                            result = {
                                "Threshold": threshold,
                                "Method": method,
                                "Run": run,
                                "Model": model_name,
                                "Accuracy": 0.0,
                                "Confusion_Matrix": [],
                                "Cohen_Kappa": 0.0,
                                "Num_Edges": num_edges,
                                "Num_Nodes": num_nodes,
                                "Modularity": modularity_score,
                                "Num_Communities": num_communities,
                                "Avg_Community_Size": avg_community_size,
                                "Avg_Clustering": avg_clustering,
                                "Graph_Density": graph_density,
                                "Avg_Degree": avg_degree
                            }
                            try:
                                logger.info(f"Training {model_name} for {progress}")
                                if model_name == "GraphSAGE":
                                    model = GraphSAGE(dim_in=x.shape[1], dim_h=128, dim_out=len(set(labels)))
                                elif model_name == "GCN":
                                    model = GCN(dim_in=x.shape[1], dim_h=128, dim_out=len(set(labels)))
                                elif model_name == "GAT":
                                    model = GAT(dim_in=x.shape[1], dim_h=16, dim_out=len(set(labels)), heads=8)
                                model = train(model, data, class_weights=class_weights)
                                acc, cm, kappa = test(model, data)
                                result.update({
                                    "Accuracy": acc,
                                    "Confusion_Matrix": cm.tolist(),
                                    "Cohen_Kappa": kappa
                                })
                                logger.info(f"Completed {progress}: Accuracy={acc:.4f}, Cohen_Kappa={kappa:.4f}, "
                                           f"Modularity={modularity_score:.4f}, Num_Communities={num_communities}, "
                                           f"Avg_Community_Size={avg_community_size:.2f}, Avg_Clustering={avg_clustering:.4f}, "
                                           f"Graph_Density={graph_density:.4f}, Avg_Degree={avg_degree:.2f}")
                            except Exception as e:
                                logger.error(f"Error in experiment for {progress}: {e}")
                                results.append(result)
                                save_results(results, threshold, method, run)
                                logger.info(f"Saved partial results to {eval_dir}")
                                pbar.update(1)
                                continue

                            results.append(result)
                            save_results(results, threshold, method, run)
                            logger.info(f"Saved results to {eval_dir}")
                            completed_evaluation.add(experiment_id)
                            checkpoint = {
                                "completed_sparsification": list(completed_sparsification),
                                "completed_evaluation": list(completed_evaluation),
                                "results": results
                            }
                            save_checkpoint(checkpoint_file, checkpoint)
                            pbar.update(1)

        # Aggregate results
        try:
            if results:
                aggregate_results(results, num_runs)
                logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
            else:
                logger.error("No results were generated.")
        except Exception as e:
            logger.error(f"Failed to aggregate results: {e}")

    except KeyboardInterrupt:
        logger.info("Run interrupted by user")
        save_checkpoint(checkpoint_file, {
            "completed_sparsification": list(completed_sparsification),
            "completed_evaluation": list(completed_evaluation),
            "results": results
        })
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        save_checkpoint(checkpoint_file, {
            "completed_sparsification": list(completed_sparsification),
            "completed_evaluation": list(completed_evaluation),
            "results": results
        })
        exit(1)