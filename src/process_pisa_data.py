"""
Download and process PISA 2022 student data for the ML project.

This script:
1. Downloads the PISA 2022 student questionnaire data (SPSS format) from OECD
2. Filters to 5 EU countries: Romania, Germany, Finland, Netherlands, Spain
3. Selects relevant columns (plausible values + feature variables)
4. Creates the target variable (top 10% in each country)
5. Saves a clean, small dataset ready for the notebook

The SPSS file is ~650 MB compressed, ~2 GB uncompressed.
Run this script once to generate data/processed/pisa_5countries.parquet

Usage:
    python src/process_pisa_data.py
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import pyreadstat

# --- Configuration ---
PISA_URL = "https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "pisa_5countries.parquet")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "pisa_5countries.csv")

COUNTRIES = {
    "ROU": "Romania",
    "DEU": "Germany",
    "FIN": "Finland",
    "NLD": "Netherlands",
    "ESP": "Spain",
}

# Plausible values (used ONLY to create the target, then dropped)
PV_COLS = ["PV1MATH", "PV1READ", "PV1SCIE"]

# Feature columns verified against the actual PISA 2022 SPSS file
FEATURE_COLS = [
    "CNT",           # Country code (3-letter ISO)
    "ST004D01T",     # Gender (1=Female, 2=Male)
    "ESCS",          # Index of economic, social and cultural status
    "HOMEPOS",       # Home possessions (WLE)
    "HISEI",         # Highest parental occupational status (0-90)
    "HISCED",        # Highest parental education (ISCED 0-6)
    "BELONG",        # Sense of belonging to school (WLE)
    "GROSAGR",       # Growth mindset (WLE)
    "MATHPERS",      # Effort and persistence in math (WLE)
    "ANXMAT",        # Math anxiety (WLE)
    "PERSEVAGR",     # Perseverance (WLE)
    "CURIOAGR",      # Curiosity (WLE)
    "FAMSUP",        # Family support (WLE)
    "MATHEFF",       # Math self-efficacy (WLE)
    "DISCLIM",       # Disciplinary climate in math lessons (WLE)
    "RELATST",       # Quality of student-teacher relationships (WLE)
    "TEACHSUP",      # Math teacher support (WLE)
    "ST062Q01TA",    # Skipped whole school day in last 2 weeks (1-4)
    "ST062Q02TA",    # Arrived late to school in last 2 weeks (1-4)
    "ST294Q02JA",    # Days/week study before school (0-7)
    "ST295Q02JA",    # Days/week study after school (0-7)
    "ST296Q04JA",    # Total homework time per week
]

ALL_COLS = PV_COLS + FEATURE_COLS


def download_pisa_data(url, raw_dir):
    """Download the PISA SPSS zip file if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "STU_QQQ_SPSS.zip")

    if os.path.exists(zip_path):
        print(f"File already exists: {zip_path}")
        return zip_path

    print(f"Downloading PISA 2022 data from {url}")
    print("This file is ~650 MB. Consider using aria2c for faster download:")
    print("  aria2c -x 16 -s 16 <url>")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb:.0f} MB ({pct:.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, zip_path, reporthook=report)
    print("\nDownload complete.")
    return zip_path


def extract_and_read(zip_path):
    """Extract the SPSS file and read only needed columns for selected countries."""
    print("Extracting zip file...")
    extract_dir = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        sav_names = [f for f in z.namelist() if f.lower().endswith(".sav")]
        if not sav_names:
            raise FileNotFoundError("No .sav file found in the zip archive")
        sav_name = sav_names[0]
        extract_path = os.path.join(extract_dir, sav_name)
        if not os.path.exists(extract_path):
            z.extract(sav_name, extract_dir)
        print(f"Extracted: {sav_name}")

    print(f"Reading {len(ALL_COLS)} columns from SPSS file...")
    df, _ = pyreadstat.read_sav(extract_path, usecols=ALL_COLS)
    print(f"Full dataset: {df.shape[0]:,} students")

    df = df[df["CNT"].isin(COUNTRIES.keys())].copy()
    print(f"After filtering to 5 countries: {df.shape[0]:,} students")

    return df


def create_target(df):
    """Create binary target: top 10% academic performers within each country."""
    df["academic_score"] = df[PV_COLS].mean(axis=1)

    thresholds = df.groupby("CNT")["academic_score"].quantile(0.90)
    print("\n90th percentile thresholds:")
    for cnt, thresh in thresholds.items():
        print(f"  {COUNTRIES[cnt]:12s}: {thresh:.1f}")

    df["top_10_percent"] = 0
    for cnt in COUNTRIES.keys():
        mask = (df["CNT"] == cnt) & (df["academic_score"] >= thresholds[cnt])
        df.loc[mask, "top_10_percent"] = 1

    print("\nTarget distribution:")
    for cnt in sorted(COUNTRIES.keys()):
        subset = df[df["CNT"] == cnt]
        pct = subset["top_10_percent"].mean() * 100
        print(f"  {COUNTRIES[cnt]:12s}: {pct:.1f}%")

    df = df.drop(columns=PV_COLS + ["academic_score"])
    return df


def sample_data(df, n_per_country=7000, random_state=42):
    """Sample up to n_per_country students per country."""
    sampled = []
    for cnt in sorted(COUNTRIES.keys()):
        subset = df[df["CNT"] == cnt]
        if len(subset) > n_per_country:
            subset = subset.sample(n=n_per_country, random_state=random_state)
        sampled.append(subset)
        print(f"  {COUNTRIES[cnt]:12s}: {len(subset):,}")

    return pd.concat(sampled, ignore_index=True)


def main():
    print("=" * 60)
    print("PISA 2022 Data Processing")
    print("=" * 60)

    zip_path = download_pisa_data(PISA_URL, RAW_DIR)
    df = extract_and_read(zip_path)
    df = create_target(df)

    print("\nSampling...")
    df = sample_data(df)
    print(f"Final: {len(df):,} students, {df.shape[1]} columns")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_PARQUET} ({os.path.getsize(OUTPUT_PARQUET)/1024/1024:.1f} MB)")
    print(f"Saved: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
