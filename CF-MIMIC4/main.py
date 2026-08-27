# -*- coding: utf-8 -*-
# name: main.py
# author: JunYoung Park
# date: 2026-08-27

from pathlib import Path
import pandas as pd

from data_preprocessing import (
    CHARTEVENT_RANGES, LABEVENT_RANGES,
    filter_age, filter_mcs, filter_cf_data,
    restrict_to_current_cohort, prepare_height_weight,
    filter_permitted_ranges, remove_numeric_duplicates,
    classify_inputevents, prepare_continuous_rate,
    map_nonpharma_concepts, merge_nonpharma_simultaneous,
    map_pharma_concepts, merge_mapped_pharma,
)

from adaptive_imputation import (
    build_time_grid, place_nonpharma_on_grid, place_pharma_on_grid, 
    calculate_imputation_parameters, adaptive_impute_nonpharma, static_variables,
)

from state_annotation import annotate_circulatory_state
from labeling import label_future_cf


# -----------------------------------------------------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------------------------------------------------
DATASET_ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\MIMIC4_subset")
MIMIC_VARS_PATH = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\etc\mimic_vars.csv")
TABLE4_PATH = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\etc\Supplementary_TABLE4.xlsx")


def main():
    # -------------------------------------------------------------------------------------------------------------------
    # 0. Load data
    # -------------------------------------------------------------------------------------------------------------------
    print("[0] Loading data...")

    stays = pd.read_csv(DATASET_ROOT / "icu/icustays.csv.gz",
                        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    admissions = pd.read_csv(DATASET_ROOT / "hosp/admissions.csv.gz",
                             usecols=["subject_id", "hadm_id", "admittime", "admission_type"])
    patients = pd.read_csv(DATASET_ROOT / "hosp/patients.csv.gz",
                           usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
    services = pd.read_csv(DATASET_ROOT / "hosp/services.csv.gz",
                           usecols=["hadm_id", "transfertime", "curr_service"])
    d_items = pd.read_csv(DATASET_ROOT / "icu/d_items.csv.gz",
                          usecols=["itemid", "label", "linksto"])
    chartevents = pd.read_csv(DATASET_ROOT / "icu/chartevents.csv.gz",
                              usecols=["stay_id", "itemid", "charttime", "valuenum"])
    labevents = pd.read_csv(DATASET_ROOT / "hosp/labevents.csv.gz",
                            usecols=["hadm_id", "itemid", "charttime", "valuenum"])
    procedureevents = pd.read_csv(DATASET_ROOT / "icu/procedureevents.csv.gz",
                                  usecols=["stay_id", "itemid"])
    inputevents = pd.read_csv(
        DATASET_ROOT / "icu/inputevents.csv.gz",
        usecols=["stay_id", "itemid", "starttime", "endtime", "rate", "rateuom",
                 "amount", "amountuom", "ordercategorydescription"]
    )

    # -------------------------------------------------------------------------------------------------------------------
    # 1. Cohort filtering
    # -------------------------------------------------------------------------------------------------------------------
    print("[1] Cohort filtering...")

    stays = filter_age(stays, admissions, patients)
    stays = filter_mcs(stays, d_items, chartevents, procedureevents)
    stays = filter_cf_data(stays, chartevents)

    # -------------------------------------------------------------------------------------------------------------------
    # 2. Measurement preprocessing
    # -------------------------------------------------------------------------------------------------------------------
    print("[2] Measurement preprocessing...")

    chartevents, labevents = restrict_to_current_cohort(stays, chartevents, labevents)
    height_weight_events = prepare_height_weight(stays, chartevents)

    chartevents = filter_permitted_ranges(chartevents, CHARTEVENT_RANGES)
    labevents = filter_permitted_ranges(labevents, LABEVENT_RANGES)

    chartevents = remove_numeric_duplicates(chartevents, id_col="stay_id")
    labevents = remove_numeric_duplicates(labevents, id_col="hadm_id")

    # -------------------------------------------------------------------------------------------------------------------
    # 3. Pharmaceutical preprocessing
    # -------------------------------------------------------------------------------------------------------------------
    print("[3] Pharmaceutical preprocessing...")

    inputevents = classify_inputevents(stays, inputevents, d_items)
    continuous = prepare_continuous_rate(inputevents)

    # -------------------------------------------------------------------------------------------------------------------
    # 4. Variable mapping / merging
    # -------------------------------------------------------------------------------------------------------------------
    print("[4] Variable mapping / merging...")

    nonpharma = map_nonpharma_concepts(stays, chartevents, labevents)
    nonpharma_merged = merge_nonpharma_simultaneous(nonpharma)

    pharma_map = map_pharma_concepts(
        inputevents,
        mimic_vars_path=MIMIC_VARS_PATH,
        table4_path=TABLE4_PATH
    )
    pharma_merged = merge_mapped_pharma(continuous, pharma_map)

    # -------------------------------------------------------------------------------------------------------------------
    # 5. 60-min grid + adaptive imputation
    # -------------------------------------------------------------------------------------------------------------------
    print("[5] Adaptive imputation...")

    time_grid, grid_stay_info = build_time_grid(stays, nonpharma_merged, grid_minutes=60)
    raw_grid_values = place_nonpharma_on_grid(nonpharma_merged, grid_stay_info, grid_minutes=60)
    pharma_grid = place_pharma_on_grid(pharma_merged, time_grid, grid_stay_info, grid_minutes=60)

    imputation_params = calculate_imputation_parameters(nonpharma_merged, grid_stay_info)

    imputed_nonpharma = adaptive_impute_nonpharma(
        time_grid=time_grid,
        raw_grid_values=raw_grid_values,
        nonpharma_merged=nonpharma_merged,
        grid_stay_info=grid_stay_info,
        imputation_params=imputation_params,
        height_weight_events=height_weight_events,
        grid_minutes=60
    )

    static_features = static_variables(
        stays, admissions, patients, services, height_weight_events
    )

    dynamic_grid = imputed_nonpharma.merge(
        pharma_grid, on=["stay_id", "gridtime"], how="left", validate="one_to_one"
    )

    # -------------------------------------------------------------------------------------------------------------------
    # 6. Circulatory state annotation
    # -------------------------------------------------------------------------------------------------------------------
    print("[6] State annotation...")

    state = annotate_circulatory_state(
        time_grid=time_grid,
        raw_grid_values=raw_grid_values,
        nonpharma_merged=nonpharma_merged,
        pharma_merged=pharma_merged
    )
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # 7. Future Circulatory Failure Labeling
    # -------------------------------------------------------------------------------------------------------------------
    print("[7] Future CF labeling...")
    
    labels = label_future_cf(state=state, horizon_hours=8.0)


    # -------------------------------------------------------------------------------------------------------------------
    # Result / Refactoring check
    # -------------------------------------------------------------------------------------------------------------------
    counts = state["state"].value_counts()
    no_cf = int(counts.get("NO_CF", 0))
    cf = int(counts.get("CF", 0))
    ambiguous = int(counts.get("AMBIGUOUS", 0))

    final_stays = int(stays["stay_id"].nunique())
    grid_stays = int(time_grid["stay_id"].nunique())
    total_points = len(state)
    cf_stays = int(state.loc[state["state"] == "CF", "stay_id"].nunique())
    
    valid_labels = labels["label"].dropna()
    positive = int((valid_labels == 1).sum())
    negative = int((valid_labels == 0).sum())

    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)
    print(f"Final stays      : {final_stays:,}")
    print(f"Grid stays       : {grid_stays:,}")
    print(f"Total grid points: {total_points:,}")

    print("\n[State Annotation]")
    print(f"NO_CF     : {no_cf:>7,} ({100 * no_cf / total_points:5.2f}%)")
    print(f"CF        : {cf:>7,} ({100 * cf / total_points:5.2f}%)")
    print(f"AMBIGUOUS : {ambiguous:>7,} ({100 * ambiguous / total_points:5.2f}%)")
    print(f"CF stays  : {cf_stays:,}")
    
    print("\n[Future CF Label]")
    print(f"Valid samples : {len(valid_labels):,}")
    print(f"Positive      : {positive:,} ({100 * positive / len(valid_labels):.2f}%)")
    print(f"Negative      : {negative:,} ({100 * negative / len(valid_labels):.2f}%)")
    print(f"Excluded      : {labels['label'].isna().sum():,}")

if __name__ == "__main__":
    main()