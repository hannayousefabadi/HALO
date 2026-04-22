import pandas as pd
import numpy as np


class FeatureMapper:
    
    """
    A class to concatenate feature vectors for drug combinations.
    """

    def __init__(self):
        pass

    
    def _validate_features_df(self, features_df):
        """
        A small utility method to check feature sets before mapping.
        """
        if 'inchikey' not in features_df.columns:
            raise ValueError("features_df must contain an 'inchikey' column.")
        
        if features_df['inchikey'].duplicated().any():
            dups = features_df[features_df['inchikey'].duplicated()]['inchikey'].tolist()
            raise ValueError(
                f"Duplicate inchikeys found in features_df: {dups[:10]}..."
            )

        ignore = {'inchikey', 'drug', 'level'}
        feature_cols = [c for c in features_df.columns if c not in ignore]

        non_numeric = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(features_df[c])
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric feature columns detected: {non_numeric[:10]}..."
            )


        
    def concatenation(self, combinations_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate features for two drugs in a combination
        Feature columns are named as:
            A_<orig_feature_name> for drug A
            B_<orig_feature_name> for drug B

        Args:
            combinations_df: drug combinations.
            features_df: contains each drugs feature set. one row, one feature set.

        Return:
            concatenated feature vectors from two drugs added to the initial combinations_df as new columns
        """
        self._validate_features_df(features_df)

        features_cols = [c for c in features_df if c not in ['drug', 'inchikey', 'level']]
        concatenated_vectors = []

        for index, row in combinations_df.iterrows():
            # sorting drug a and drug b based on their Inchikeys, meaning drug a is always the one above drug b in the Inchikey order
            drug_a, drug_b = sorted([row['Drug A Inchikey'], row['Drug B Inchikey']])
            # drug_a = row['Drug A Inchikey'] 
            # drug_b = row['Drug B Inchikey']

            drug_a_features = features_df[features_df['inchikey'] == drug_a][features_cols].to_numpy().flatten()
            drug_b_features = features_df[features_df['inchikey'] == drug_b][features_cols].to_numpy().flatten()

            concatenated_vector = np.concatenate([drug_a_features, drug_b_features])
            concatenated_vectors.append(concatenated_vector)

        total_len = len(concatenated_vector)
        # build interpretable column names
        a_cols = [f"A_{c}" for c in features_cols]
        b_cols = [f"B_{c}" for c in features_cols]
        col_names = a_cols + b_cols

        features_df_final = pd.DataFrame(
            concatenated_vectors,
            columns=col_names,
            index=combinations_df.index
        )
        final_df = pd.concat([combinations_df, features_df_final], axis=1)

        return final_df


    def elementwise_similarity(self, combinations_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute element-wise Cosine similarity and Euclidean-derived similarity (1 − normalized distance)
        (2 * 128 features per level).

        - cos_elem[i]  = (x/||x||)_i * (y/||y||)_i      in [-1, 1]
        - euc_elem[i]  = 1 - |x_i - y_i|                in [-1, 1]  (deterministic scaling)

        Args:
            combinations_df
            features_df

        Return:
            returns 256 features per level (per drug pair), 6400 features overall
            
        """
        self._validate_features_df(features_df)

        eps = 1e-8
        features_cols = [c for c in features_df if c not in['drug', 'inchikey', 'level']]
        result = []

        for index, row in combinations_df.iterrows():
            # sorting drug a and drug b based on their inchikeys, meaning drug a is always the one above drug b in the inchikey order
            drug_a, drug_b = sorted([row['Drug A Inchikey'], row['Drug B Inchikey']])
            
            drug_a_features = features_df[features_df['inchikey'] == drug_a][features_cols].to_numpy().flatten()
            drug_b_features = features_df[features_df['inchikey'] == drug_b][features_cols].to_numpy().flatten()

            # Cosine contributions per-dimension (Hadamard of normalized vectors)
            drug_a_norm = drug_a_features / (np.linalg.norm(drug_a_features) + eps)
            drug_b_norm = drug_b_features / (np.linalg.norm(drug_b_features) + eps)
            cos_elem = drug_a_norm * drug_b_norm # scaled Cosine similarity to [-1, 1]

            euc_elem = 1.0 - np.abs(drug_a_features - drug_b_features) # scaling Euclidan to [-1, 1]
           
            combined = np.concatenate([cos_elem, euc_elem])
            result.append(combined)

        total_len = len(result[0])
        col_names = [f'cos_elem_{i}' for i in range(total_len // 2)] + [f'euc_elem_{i}' for i in range(total_len // 2)]
        features_df_final = pd.DataFrame(result, columns=col_names, index=combinations_df.index)
        return pd.concat([combinations_df, features_df_final], axis=1)


    def compact_similarity(self, combinations_df: pd.DataFrame, features_df: pd.DataFrame, block_size: int = 128) -> pd.DataFrame:
        """
        Compute compact similarity set: one Cosine similarity per sublevel, one Euclidean-derived similarity (1 − normalized distance) per sublevel

        - cos_block[l] = Cosine over sublevel 1              in [-1, 1]
        - euc_block[l] = 1 - ||a_l - b_l||/(2*sqrt(k))       in [-1, 1] 

        """
        self._validate_features_df(features_df)
        
        eps = 1e-8
        features_cols = [c for c in features_df if c not in ['drug', 'inchikey', 'level']]
        result = []

        n_feats = len(features_cols)
        if n_feats % block_size != 0:
            raise ValueError(f"Feature length ({n_feats}) is not divisible by block_size ({block_size}). "
                             "Set block_size correctly for your current features_df.")

        n_blocks = n_feats // block_size
        max_l2_block = 2.0 * np.sqrt(block_size)  # deterministic bound for CC [-1,1]


        for index, row in combinations_df.iterrows():
            # sorting drug a and drug b based on their inchikeys, meaning drug a is always the one above drug b in the inchikey order
            drug_a, drug_b = sorted([row['Drug A Inchikey'], row['Drug B Inchikey']])

            drug_a_features = features_df[features_df['inchikey'] == drug_a][features_cols].to_numpy().flatten()
            drug_b_features = features_df[features_df['inchikey'] == drug_b][features_cols].to_numpy().flatten()

            sims = []
            for i in range(n_blocks):
                s = i * block_size
                e = s + block_size
                a_blk, b_blk = drug_a_features[s:e], drug_b_features[s:e]

                # Cosine and Euclidean per block
                denom = (np.linalg.norm(a_blk) * np.linalg.norm(b_blk)) + eps
                cos_sim = float(np.dot(a_blk, b_blk) / denom)

                # normalized L2 per block mapped to [-1,1]
                d = float(np.linalg.norm(a_blk - b_blk))
                euc_dis = 1.0 - (d / (max_l2_block + eps))

                sims.extend([cos_sim, euc_dis])

            result.append(sims)

        col_names = [f'cos_block_{i}' for i in range(n_blocks)] + [f'euc_block_{i}' for i in range(n_blocks)]

        features_df_final = pd.DataFrame(result, columns=col_names, index=combinations_df.index)
        return pd.concat([combinations_df, features_df_final], axis=1)















