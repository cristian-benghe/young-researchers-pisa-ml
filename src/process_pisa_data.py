"""
Download and process PISA 2022 student data for the ML project.

This script:
1. Downloads the PISA 2022 student questionnaire data (SPSS format) from OECD
2. Filters to 5 EU countries: Romania, Germany, Finland, Netherlands, Spain
3. Selects relevant columns (plausible values + feature variables)
4. Creates the target variable (top 10% in each country)
5. Saves a clean, small dataset ready for the notebook

Run this script once to generate data/processed/pisa_5countries.parquet
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import pyreadstat

# --- Configuration ---
PISA_URL = "https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "pisa_5countries.parquet")

# Country codes in PISA 2022
COUNTRIES = {
    "ROU": "Romania",
    "DEU": "Germany",
    "FIN": "Finland",
    "NLD": "Netherlands",
    "ESP": "Spain",
}

# Plausible values (used ONLY to create the target, then dropped)
PV_COLS = ["PV1MATH", "PV1READ", "PV1SCIE"]

# Feature columns we want to keep
# Each is explained in the notebook
FEATURE_COLS = [
    "CNT",          # Country code (3-letter ISO)
    "ST004D01T",    # Gender (1=Female, 2=Male)
    "ESCS",         # Index of economic, social and cultural status
    "HOMEPOS",      # Home possessions index
    "ICTRES",       # ICT resources at home
    "WEALTH",       # Family wealth index
    "HEDRES",       # Home educational resources
    "CULTPOSS",     # Cultural possessions at home
    "HISEI",        # Highest parental occupational status
    "HISCED",       # Highest parental education (ISCED level)
    "MMINS",        # Learning time in math (minutes per week)
    "LMINS",        # Learning time in language (minutes per week)
    "SMINS",        # Learning time in science (minutes per week)
    "BELONG",       # Sense of belonging to school
    "MASTGOAL",     # Mastery goal orientation (growth mindset)
    "GFOFAIL",      # Fear of failure
    "SWBP",         # Subjective well-being (positive affect)
    "ATTLNACT",     # Attitude toward learning activities
    "EMOSUPS",      # Emotional support from parents
    "ST062Q01TA",   # Skipped whole school day in last 2 weeks
    "ST062Q02TA",   # Arrived late in last 2 weeks
    "ST250Q01JA",   # Access to computer at home for schoolwork
]

# All columns we need to read from the SPSS file
ALL_COLS = PV_COLS + FEATURE_COLS


def download_pisa_data(url, raw_dir):
    """Download the PISA SPSS zip file if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "STU_QQQ_SPSS.zip")

    if os.path.exists(zip_path):
        print(f"File already exists: {zip_path}")
        return zip_path

    print(f"Downloading PISA 2022 data from {url}")
    print("This file is ~1.5 GB and may take several minutes...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    return zip_path


def extract_and_read(zip_path, countries, columns):
    """Extract the SPSS file and read only needed columns for selected countries."""
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as z:
        sav_names = [f for f in z.namelist() if f.endswith(".sav")]
        if not sav_names:
            raise FileNotFoundError("No .sav file found in the zip archive")
        sav_name = sav_names[0]
        extract_path = os.path.join(os.path.dirname(zip_path), sav_name)
        if not os.path.exists(extract_path):
            z.extract(sav_name, os.path.dirname(zip_path))
        print(f"Extracted: {extract_path}")

    print("Reading SPSS file (this takes a moment)...")
    # Read only the columns we need, filtering by country
    df, meta = pyreadstat.read_sav(
        extract_path,
        usecols=["CNT"] + columns,
    )
    print(f"Full dataset shape: {df.shape}")

    # Filter to our 5 countries
    df = df[df["CNT"].isin(countries.keys())].copy()
    print(f"After filtering to 5 countries: {df.shape}")

    return df


def create_target(df):
    """
    Create the binary target variable.
    - academic_score = mean of PV1MATH, PV1READ, PV1SCIE
    - top_10_percent = 1 if score >= 90th percentile WITHIN that student's country
    """
    df["academic_score"] = df[PV_COLS].mean(axis=1)

    # Calculate 90th percentile per country
    thresholds = df.groupby("CNT")["academic_score"].quantile(0.90)
    print("\n90th percentile thresholds by country:")
    for cnt, thresh in thresholds.items():
        print(f"  {COUNTRIES.get(cnt, cnt)}: {thresh:.1f}")

    # Create binary target
    df["top_10_percent"] = 0
    for cnt in df["CNT"].unique():
        mask = df["CNT"] == cnt
        threshold = thresholds[cnt]
        df.loc[mask & (df["academic_score"] >= threshold), "top_10_percent"] = 1

    # Verify proportions
    print("\nTarget distribution by country:")
    for cnt in sorted(df["CNT"].unique()):
        subset = df[df["CNT"] == cnt]
        pct = subset["top_10_percent"].mean() * 100
        print(f"  {COUNTRIES.get(cnt, cnt)}: {pct:.1f}% in top 10%")

    # Drop the score columns (they must NOT be used as features)
    df = df.drop(columns=PV_COLS + ["academic_score"])

    return df


def sample_data(df, n_per_country=7000, random_state=42):
    """Sample up to n_per_country students per country for manageable size."""
    sampled = []
    for cnt in df["CNT"].unique():
        subset = df[df["CNT"] == cnt]
        if len(subset) > n_per_country:
            subset = subset.sample(n=n_per_country, random_state=random_state)
        sampled.append(subset)
    result = pd.concat(sampled, ignore_index=True)
    print(f"\nFinal dataset after sampling: {result.shape}")
    return result


def main():
    # Step 1: Download
    zip_path = download_pisa_data(PISA_URL, RAW_DIR)

    # Step 2: Read and filter
    df = extract_and_read(zip_path, COUNTRIES, ALL_COLS[1:])  # CNT already included

    # Step 3: Create target variable
    df = create_target(df)

    # Step 4: Sample for manageable size
    df = sample_data(df)

    # Step 5: Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved processed dataset to: {OUTPUT_FILE}")
    print(f"Final columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Also save as CSV for maximum compatibility
    csv_path = OUTPUT_FILE.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Also saved as CSV: {csv_path}")


if __name__ == "__main__":
    main()
