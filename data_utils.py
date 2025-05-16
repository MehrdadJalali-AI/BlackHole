import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import logging

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)

def load_edges_list(filename):
    try:
        edges = pd.read_csv(filename)
        if not all(col in edges.columns for col in ['source', 'target', 'weight']):
            raise ValueError("Edges CSV must contain 'source', 'target', 'weight' columns")
        return edges
    except Exception as e:
        logger.error(f"Failed to load edges from {filename}: {e}")
        raise

def get_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return np.random.randn(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to process SMILES {smiles}: {e}")
        return np.random.randn(n_bits)

def load_summary_data(filename, node_labels):
    try:
        summary_data = pd.read_csv(filename, index_col=0)
        required_columns = ['Pore Limiting Diameter', 'linker SMILES', 'metal', 'Largest Cavity Diameter', 'Largest Free Sphere']
        if not all(col in summary_data.columns for col in required_columns):
            raise ValueError(f"Summary CSV missing required columns: {required_columns}")
        
        # Data cleaning
        summary_data = summary_data[summary_data.index.isin(node_labels)].copy()
        numeric_cols = ['Pore Limiting Diameter', 'Largest Cavity Diameter', 'Largest Free Sphere']
        for col in numeric_cols:
            summary_data[col] = pd.to_numeric(summary_data[col], errors='coerce')
            nan_count = summary_data[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in {col}, filling with median")
                summary_data[col] = summary_data[col].fillna(summary_data[col].median())
        summary_data['linker SMILES'] = summary_data['linker SMILES'].replace('F[Si](F)(F)(F)(F)F', 'c1ccccc1').fillna('c1ccccc1')
        summary_data['metal'] = summary_data['metal'].fillna('Cu')
        
        # Log data summary
        logger.info(f"Summary data rows: {len(summary_data)}, unique metals: {summary_data['metal'].nunique()}")
        
        # Generate features
        features = []
        for idx, row in summary_data.iterrows():
            smiles = row['linker SMILES']
            fp = get_morgan_fingerprint(smiles)
            other_features = np.array([
                row['Pore Limiting Diameter'],
                row['Largest Cavity Diameter'],
                row['Largest Free Sphere']
            ], dtype=np.float32)
            if other_features.shape != (3,) or np.any(np.isnan(other_features)):
                logger.error(f"Invalid other_features for node {idx}: shape {other_features.shape}, values {other_features}")
                other_features = np.array([summary_data[col].median() for col in numeric_cols], dtype=np.float32)
            metal = row['metal']
            metal_one_hot = np.zeros(4, dtype=np.float32)  # [Cu, Zn, Fe, Co]
            metal_map = {'Cu': 0, 'Zn': 1, 'Fe': 2, 'Co': 3}
            if metal in metal_map:
                metal_one_hot[metal_map[metal]] = 1.0
            else:
                logger.warning(f"Unknown metal {metal} for node {idx}, defaulting to Cu")
                metal_one_hot[0] = 1.0
            logger.debug(f"Node {idx}: fp shape {fp.shape}, other_features shape {other_features.shape}, metal_one_hot shape {metal_one_hot.shape}")
            feature = np.concatenate([fp, other_features, metal_one_hot])
            if feature.shape != (1031,):
                logger.error(f"Feature shape mismatch for node {idx}: {feature.shape}, fp {fp.shape}, other_features {other_features.shape}, metal_one_hot {metal_one_hot.shape}")
                raise ValueError(f"Feature shape mismatch for node {idx}")
            features.append(feature)
        
        features_df = pd.DataFrame(features, index=summary_data.index)
        logger.info(f"Generated features with shape {features_df.shape}")
        return features_df, summary_data
    except Exception as e:
        logger.error(f"Failed to load summary data from {filename}: {e}")
        raise