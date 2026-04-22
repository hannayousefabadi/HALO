import pandas as pd

class DrugMapper:

    """ 
    This class will be preprocessing the combination datasets, including cleaning the datasets, 
    mapping inchikeys from list_antibacterial and list_antivirals to the training sets, removing the 
    NA or repetitive rows, checking if their drugs are present in the chemicalchecker database feature sets,
    and 
    """

    def __init__(self):
        pass


    def inspect_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Turning all the drug names lower case, also getting initial info of the data
        """
        df[['Drug A', 'Drug B']] = df[['Drug A', 'Drug B']].apply(lambda col: col.str.lower())
        df.columns = df.columns.str.strip()

        all_compounds = pd.concat([df['Drug A'], df['Drug B']]).drop_duplicates().tolist()

        print(f'number of initial combinations in this dataset: {len(df)}')
        print(f'number of unique antibacterial/antiviral compounds in this dataset:{len(all_compounds)}')

        return df


    def compounds_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Getting a list of all the antibacterials/antivirals in this dataset
        """
        if hasattr(self, 'all_compounds'):
            return sorted(self.all_compounds)
        else:
            import pandas as pd
            all_compounds = pd.concat([df['Drug A'], df['Drug B']]).drop_duplicates().tolist()
            return (sorted(all_compounds))



    def enrich(self, df: pd.DataFrame, lookup_df: pd.DataFrame, col1='Drug A', col2='Drug B') -> pd.DataFrame:
        """
        Appending  inchikeys column to combination DataFrame
        """
        lookup = dict(zip(lookup_df['drug'], lookup_df['inchikey']))
        df[f'{col1} Inchikey'] = df[col1].map(lookup)
        df[f'{col2} Inchikey'] = df[col2].map(lookup)
        df['Drug A Inchikey'] = df['Drug A Inchikey'].astype(str).str.strip().str.upper()
        df['Drug B Inchikey'] = df['Drug B Inchikey'].astype(str).str.strip().str.upper()
        return df
    


    def missing_compounds(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Getting compounds with missing inchikeys, AKA the number of compounds
        not included in the initial `list_antimicrobial` or `list_antiviral`
        
        """
        # Returning the number of missing inchikeys in the whole dataset (AKA rows with either
        # Drug A inchikey or Drug B inchikey is missing)
        num_missing_A = df['Drug A Inchikey'].isna().sum()
        num_missing_B = df['Drug B Inchikey'].isna().sum()
        
        missing_A = df.loc[df['Drug A Inchikey'].isna(), 'Drug A']
        missing_B = df.loc[df['Drug B Inchikey'].isna(), 'Drug B']
        missing_compounds = pd.concat([missing_A, missing_B]).drop_duplicates()
        missing_compounds = missing_compounds.rename('missing drugs')
        # Saving this list as a .csv
        # filename = f'{dataset_name}_missing_compounds.csv'
        # missing_compounds.to_csv(filename, index=False)

        print(f'missing Drug A Inchikeys: {num_missing_A}')
        print(f'missing Drug B Inchikeys: {num_missing_B}')
        return sorted(missing_compounds)



 
    # Preprocessing data for modeling; handling the NA values in all the critical columns
    def check_na(self, df: pd.DataFrame, critical_columns: list[str]) -> pd.DataFrame:
        """
        Check for NA values in the columns that are critical for modeling.

        Args:
            df: dataframe to clean,
            critical_columns: list of column names to check NA

        Return:
            Cleaned df: dataset without NA,
            na_report: A series with missing values counts for each column    
        
        """
        df = df.copy()
        na_count = df[critical_columns].isna().sum()
        na_report = na_count[na_count > 0]

        if not na_report.empty:
            print(f'Missing values report (before dropping): {na_report}')
        else:
            print('No missing rows found in the critical columns.')


        cleaned_df = df.dropna(subset=critical_columns)

        return cleaned_df 


    # Get the cleaned dataset with no NA and no repeatitions
    def refine_combinations(self, df: pd.DataFrame, other_columns: list[str], bliss_round=4) -> pd.DataFrame:
        """
        Refine the combination dataframes by:
        * removing the NAs in Drug A or Drug B inchikeys
        * removing repeated redundant rows (drug combination, organism, method used, value, etc.)

        Args:
            df: pd.DataFarme to clean
            other_columns (list): additional columns used to identified duplicated rows

        Retunrs:
        
        """
        # Eliminating duplicated rows with same drug pairs + same associated metadata (organism,
        # method, value, etc.) 
        # e.g. the drug pair can be swapped: tetracycline + vancomycin Vs. vancomycin + tetracycline)

        df = df.copy()

        # taking care of the bliss score, convert to numeric and rounding to 4 digits:
        if 'Bliss Score' in df.columns:
            df['Bliss Score'] = pd.to_numeric(df['Bliss Score'], errors='coerce')
            df['Bliss Score'] = df['Bliss Score'].round(bliss_round)

        # Creating a column with tuples for data combinations
        df['Drug Pair'] = df.apply(lambda x: tuple(sorted([x['Drug A Inchikey'], x['Drug B Inchikey']])), axis=1)
        depute_key = ['Drug Pair'] + other_columns
        duplicates_mask = df.duplicated(subset=depute_key, keep='first')
        duplicates = df.loc[duplicates_mask]

        df_cleaned = df.loc[~duplicates_mask].copy()

        len_df_cleaned = len(df_cleaned)
        print(f'numebr of repeated row: {len(duplicates)}')
        print(f'duplicated rows: {duplicates}')
        return df_cleaned




    def filter_cc_missing(self, features_df: pd.DataFrame, combinations_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method will look up drugs from combination dataset to check if they were present in the chemicalchecker database
        or not. if Drug A or Drug B is not present in the data, the whole row will be eliminated from the combination set.

        Args:
            features_df: the chemical checker data, with one row for each drug (25 levels into one row)
            combinations_df: the combination dataset with 'Drug A' and 'Drug B' as combinations in each row

        Returns:
            cleaned_combination_df: only contains drugs that are present in the chemicalchecker database. same structure as the initial,
            combinations_df
        """
        cc_drug = features_df['inchikey'].to_numpy()
        missing_drugs = set()

        mask = combinations_df['Drug A Inchikey'].isin(cc_drug) & combinations_df['Drug B Inchikey'].isin(cc_drug)
        cleaned_combination_df = combinations_df[mask].reset_index(drop=True)

        dropped_rows = combinations_df[~mask]
        for index, row in dropped_rows.iterrows():
            drug_a = row['Drug A Inchikey']
            drug_b = row['Drug B Inchikey']

            if drug_a not in cc_drug:
                missing_drugs.add(drug_a)
            if drug_b not in cc_drug:
                missing_drugs.add(drug_b)

        missing_drugs = sorted(missing_drugs)

        print(f'The number of final combinations is: {len(cleaned_combination_df)}')
        print(f'The inchikeys missing from feature set are: {missing_drugs}')
        return cleaned_combination_df

















