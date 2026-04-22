#!/usr/bin/env python3

"""
Fetch Chemical Checker signature vectors (128-dim) for a curated antibacterial drug list 
across levels A1–E5 via the ChemicalChecker API, with retries and status tracking, and 
save the resulting long-format table to 
`data/features/chemicalchecker_cc/chemicalchecker_data.csv`
"""


import pandas as pd
import requests
import time

from halo.paths import DRUG_LISTS, CC_FEATURES

INPUT_PATH = DRUG_LISTS / "list_antibacterial_for_cc.csv"
OUTPUT_PATH = CC_FEATURES / "cc_features_raw.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

list_antibacterial = pd.read_csv(INPUT_PATH).copy()
inchikeys = list_antibacterial['inchikey'].tolist()
drugs = list_antibacterial['drug'].tolist()

base_url = "https://chemicalchecker.com/api/db/getSignature"
levels = ['A1', 'A2', 'A3', 'A4', 'A5', 
            'B1', 'B2', 'B3', 'B4', 'B5', 
            'C1', 'C2', 'C3', 'C4', 'C5', 
            'D1', 'D2', 'D3', 'D4', 'D5', 
            'E1', 'E2', 'E3', 'E4', 'E5']

# The goal here is to create a final pd.DataFrame that stores these columns: drug name,
# inchikey, signature level (A1 through E5), a 128dimensional vector os sig2 data, in the
# next 128 columns

# Creating a list of all the data we have fetched. this list will eventually trun into a df.
# with each of it's elements as a row in the final DataFrame.
data_rows = []

def fetch_data(drug: str, inchikey: str, level: str) -> dict:
    """
    Main data fetching function. to get the data vector in one try, it will either end up with a proper response
    from the API, or it will face the "bad_status" or "connection_error". these two errors will be addressed in the next
    function and the use of an iteration loop in the end.

    Args:
        drug: one drug name
        inchikey: one inchikey, prospective to the drug
        level: one level

    Return:
        row: a dictionary for each drug, containing these key and their values: drug, inchikey, level, fetch_status, dim_0 
        through dim_127 (the 128dimensional data vector)    
    """
    row = {
            "drug": drug,
            "inchikey": inchikey,
            "level": level
            }
    url = f"{base_url}/{level}/{inchikey}" # for e.g.: chemicalchecker.com/api/db/getSignature/A1/RZVAJINKPMORJF-UHFFFAOYSA-N
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if inchikey in data: # inchikey is present in the database and we fetched the data successfully
                vector = data[inchikey]
                row.update({f"dim_{i}": val for i, val in enumerate(vector)})
                row["fetch_status"] = "success"
                print(f"Level {level} for {drug} fetched successfully✅\n")
            else: # if the inchikey is not present in the database, returning NA for that entire row
                row.update({f"dim_{i}": None for i in range(128)}) # using a placeholder for the data vector to fetch later
                row["fetch_status"] = "not_found" 
                print(f"Level {level} for {drug} not found in the ChemicalChecker databse❌\n")
        else:
            row.update({f"dim_{i}": None for i in range(128)})
            row["fetch_status"] = "bad_status"
            print(f"Server did not respond for {drug} in level {level}\n")

    except requests.exceptions.RequestException as error: # If there was any connection error that prevented the request to go through
        row.update({f"dim_{i}": None for i in range(128)}) # Using a placeholder again
        row['fetch_status'] = "connection_error"
        print(f"Connection error for {drug} in level {level}\n")

    return row
    


def retry_fetch(drug: str, inchikey: str, level: str, max_attempts: int = 4) -> dict:
    """
    Retrying fetching the data for the rows that had the "bad_status" or "connection_error" from the previous function.
    however the args and outputs of this function is the same as `fetch_data`.

    Args:
        drug: one drug name
        inchikey: one inchikey, prospective to the drug
        level: one level

    Return:
        row: a dictionary for each drug, containing these key and their values: drug, inchikey, level, fetch_status, dim_0 
        through dim_127 (the 128dimensional data vector)
    """
    attempts = 0
    row = None

    while attempts < max_attempts:
        row = fetch_data(drug, inchikey, level)

        if row["fetch_status"] in ["success", "not_found"]:
            # data_rows.append(row)
            break
        elif row["fetch_status"] in ["bad_status", "connection_error"]:
            attempts += 1
            print(f"Retrying {drug} at level {level}: attempts {attempts}/4\n")    
            time.sleep(2)

    return row


# Main for loop for iterating through combinations of drug(inchikey)-level
# The while loop is designed to retry fetching the data until all the rows status is either "success" or "not_found" 

def main():
    max_rounds = 5 # outer rounds of retry_fetch per combo (each does its own 4 attempts)
    sleep_between_rounds = 3

    for level in levels:
        for drug, inchikey in zip(drugs, inchikeys):
            rounds = 0
            while True:
                row = retry_fetch(drug, inchikey, level, max_attempts=4)
                status = row["fetch_status"]

                if status in ("success", "not_found"):
                    data_rows.append(row) # append once per finished combo
                    break

                rounds += 1
                if rounds >= max_rounds:
                    print(f"Giving up on {drug} @ {level} after {rounds} rounds of retry_fetch.")
                    data_rows.append(row)
                    break

                time.sleep(sleep_between_rounds)
    
    df = pd.DataFrame(data_rows)
    df['level'] = pd.Categorical(df['level'], categories=levels, ordered=True)
    df['drug'] = pd.Categorical(df['drug'], categories=drugs, ordered=True)
    df = df.sort_values(by=['level', 'drug']).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)



if __name__ == "__main__":
    main()
