import os
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_file):
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_sparsification": [], "completed_evaluation": [], "results": []}
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {"completed_sparsification": [], "completed_evaluation": [], "results": []}

def save_checkpoint(checkpoint_file, checkpoint):
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=4)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def save_results(results, threshold, method, run):
    try:
        eval_dir = f"evaluation/threshold_{threshold:.2f}/method_{method}/run_{run}"
        os.makedirs(eval_dir, exist_ok=True)
        output_file = os.path.join(eval_dir, "model_results_with_error_bars.csv")
        df = pd.DataFrame(results)
        df = df[(df['Threshold'] == threshold) & (df['Method'] == method) & (df['Run'] == run)]
        if df.empty:
            logger.warning(f"No results to save for threshold={threshold}, method={method}, run={run}")
        else:
            df.to_csv(output_file, index=False)
            if os.path.exists(output_file):
                logger.info(f"Saved results to {output_file} with {len(df)} rows")
            else:
                logger.error(f"Failed to create {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results for threshold={threshold}, method={method}, run={run}: {e}")

def aggregate_results(results, num_runs):
    try:
        df = pd.DataFrame(results)
        if df.empty:
            logger.warning("No results to aggregate")
            return
        metrics = ['Accuracy', 'Cohen_Kappa', 'Modularity', 'Num_Communities', 'Avg_Community_Size', 
                  'Avg_Clustering', 'Graph_Density', 'Avg_Degree']
        agg_results = df.groupby(['Threshold', 'Method', 'Model'])[metrics].agg(['mean', 'std']).reset_index()
        agg_results.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_results.columns]
        os.makedirs("evaluation_results", exist_ok=True)
        output_file = "evaluation_results/model_results_with_error_bars.csv"
        agg_results.to_csv(output_file, index=False)
        logger.info(f"Saved aggregated results to {output_file} with {len(agg_results)} rows")
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")