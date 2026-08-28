# -*- coding: utf-8 -*-
# name: main.py
# author: JunYoung Park
# date: 2026-08-28

"""
    [0] Load data
    [1] Corhort filtering
    [2] Data split
    [3] Measurement preprocessing       ← FULL
    [4] Pharmaceutical preprocessing    ← FULL
    [5] Variable mapping / merging      ← FULL
    [6] Time grid 생성                  ← FULL
    [7] State annotation                ← FULL
    [8] Future CF labeling              ← FULL  
    [9] Imputation                      ← Split
    [10] Feature generation             ← Split
    [11] ...
"""

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
from data_split import build_subject_splits, get_split_stay_ids, summarize_splits
from adaptive_imputation import (
    build_time_grid, place_nonpharma_on_grid, place_pharma_on_grid, 
    calculate_imputation_parameters, adaptive_impute_nonpharma, static_variables,
)
from state_annotation import annotate_circulatory_state
from labeling import label_future_cf
from feature_generation import (
    prepare_feature_grid,
    add_static_features,
    extract_multiresolution_features,
    extract_instability_history_features,
    extract_measurement_intensity_features,
    extract_shapelet_features,
    extract_time_since_admission,
    make_feature_csv,
)
from model_training import run_lightgbm_training

# -----------------------------------------------------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------------------------------------------------
DATASET_ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\MIMIC4_subset")
MIMIC_VARS_PATH = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\etc\mimic_vars.csv")
TABLE4_PATH = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets\etc\Supplementary_TABLE4.xlsx")
FEATURE_OUTPUT_ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\CF-MIMIC4")
MODEL_OUTPUT_ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\CF-MIMIC4model_results")


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
    # 2. Data split
    # -------------------------------------------------------------------------------------------------------------------
    print("[2] Data split...")

    splits = build_subject_splits(
        stays=stays,
        n_splits=5,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42
    )

    print(summarize_splits(splits))

    # -------------------------------------------------------------------------------------------------------------------
    # 3. Measurement preprocessing
    # -------------------------------------------------------------------------------------------------------------------
    print("[3] Measurement preprocessing...")

    chartevents, labevents = restrict_to_current_cohort(stays, chartevents, labevents)
    height_weight_events = prepare_height_weight(stays, chartevents)

    chartevents = filter_permitted_ranges(chartevents, CHARTEVENT_RANGES)
    labevents = filter_permitted_ranges(labevents, LABEVENT_RANGES)

    chartevents = remove_numeric_duplicates(chartevents, id_col="stay_id")
    labevents = remove_numeric_duplicates(labevents, id_col="hadm_id")

    # -------------------------------------------------------------------------------------------------------------------
    # 4. Pharmaceutical preprocessing
    # -------------------------------------------------------------------------------------------------------------------
    print("[4] Pharmaceutical preprocessing...")

    inputevents = classify_inputevents(stays, inputevents, d_items)
    continuous = prepare_continuous_rate(inputevents)

    # -------------------------------------------------------------------------------------------------------------------
    # 5. Variable mapping / merging
    # -------------------------------------------------------------------------------------------------------------------
    print("[5] Variable mapping / merging...")

    nonpharma = map_nonpharma_concepts(stays, chartevents, labevents)
    nonpharma_merged = merge_nonpharma_simultaneous(nonpharma)

    pharma_map = map_pharma_concepts(
        inputevents,
        mimic_vars_path=MIMIC_VARS_PATH,
        table4_path=TABLE4_PATH
    )
    pharma_merged = merge_mapped_pharma(continuous, pharma_map)

    # -------------------------------------------------------------------------------------------------------------------
    # 6. 60-min time grid
    # -------------------------------------------------------------------------------------------------------------------
    print("[6] Time grid...")

    time_grid, grid_stay_info = build_time_grid(stays, nonpharma_merged, grid_minutes=60)
    raw_grid_values = place_nonpharma_on_grid(nonpharma_merged, grid_stay_info, grid_minutes=60)
    pharma_grid = place_pharma_on_grid(pharma_merged, time_grid, grid_stay_info, grid_minutes=60)

    # -------------------------------------------------------------------------------------------------------------------
    # 7. Circulatory state annotation
    # -------------------------------------------------------------------------------------------------------------------
    print("[7] State annotation...")

    state = annotate_circulatory_state(
        time_grid=time_grid,
        raw_grid_values=raw_grid_values,
        nonpharma_merged=nonpharma_merged,
        pharma_merged=pharma_merged
    )


    # -------------------------------------------------------------------------------------------------------------------
    # 8. Future Circulatory Failure Labeling
    # -------------------------------------------------------------------------------------------------------------------
    print("[8] Future CF labeling...")

    labels = label_future_cf(state=state, horizon_hours=8.0)

    # -------------------------------------------------------------------------------------------------------------------
    # 9. Split-dependent adaptive imputation
    # -------------------------------------------------------------------------------------------------------------------
    print("[9] Adaptive imputation...")

    split_data = {}

    for split_id in range(1, 6):
        print(f"  - Split {split_id}")

        train_ids, val_ids, test_ids = get_split_stay_ids(
            splits=splits,
            split_id=split_id
        )

        imputation_params = calculate_imputation_parameters(
            nonpharma_merged,
            grid_stay_info,
            stay_ids=train_ids
        )

        imputed_nonpharma = adaptive_impute_nonpharma(
            time_grid=time_grid,
            raw_grid_values=raw_grid_values,
            nonpharma_merged=nonpharma_merged,
            grid_stay_info=grid_stay_info,
            imputation_params=imputation_params,
            height_weight_events=height_weight_events,
            training_stay_ids=train_ids,
            grid_minutes=60
        )

        static_features = static_variables(
            stays,
            admissions,
            patients,
            services,
            height_weight_events,
            training_stay_ids=train_ids
        )

        dynamic_grid = imputed_nonpharma.merge(
            pharma_grid,
            on=["stay_id", "gridtime"],
            how="left",
            validate="one_to_one"
        )

        split_data[split_id] = {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "imputation_params": imputation_params,
            "dynamic_grid": dynamic_grid,
            "static_features": static_features,
        }

    # -------------------------------------------------------------------------------------------------------------------
    # 10. Feature generation
    # -------------------------------------------------------------------------------------------------------------------
    print("[10] Feature generation...")

    feature_report = []

    for split_id in range(1, 6):
        print(f"  - Split {split_id}")

        data = split_data[split_id]
        train_ids = data["train_ids"]
        dynamic_grid = data["dynamic_grid"]
        static_features = data["static_features"]
        imputation_params = data["imputation_params"]

        # Feature generation 대상 grid
        feature_grid = prepare_feature_grid(
            dynamic_grid=dynamic_grid,
            grid_minutes=60,
            ignore_first_minutes=30
        )

        # 1. Static
        static_feature_grid = add_static_features(
            feature_grid=feature_grid,
            static_features=static_features
        )

        # 2. Multi-resolution
        multiresolution_features = extract_multiresolution_features(
            dynamic_grid=dynamic_grid,
            feature_grid=feature_grid,
            imputation_params=imputation_params,
            table4_path=TABLE4_PATH,
            grid_minutes=60
        )

        # 3. Instability history
        instability_features = extract_instability_history_features(
            dynamic_grid=dynamic_grid,
            feature_grid=feature_grid,
            static_features=static_features,
            grid_minutes=60
        )

        # 4. Measurement intensity
        measurement_features = extract_measurement_intensity_features(
            raw_grid_values=raw_grid_values,
            dynamic_grid=dynamic_grid,
            feature_grid=feature_grid,
            grid_minutes=60
        )

        # 5. Shapelet
        shapelet_features, selected_shapelets = extract_shapelet_features(
            nonpharma_merged=nonpharma_merged,
            state=state,
            feature_grid=feature_grid,
            train_ids=train_ids,
            imputation_params=imputation_params,
            grid_minutes=5,
            n_cases=300,
            n_controls=300,
            n_shapelets=20,
            seed=42 + split_id
        )

        # 6. Time since admission
        time_features = extract_time_since_admission(
            dynamic_grid=dynamic_grid,
            feature_grid=feature_grid
        )

        # Feature blocks merge
        features = static_feature_grid

        for feature_block in [
            multiresolution_features,
            instability_features,
            measurement_features,
            shapelet_features,
            time_features
        ]:
            features = features.merge(
                feature_block,
                on=["stay_id", "gridtime"],
                how="left",
                validate="one_to_one"
            )

        features = make_feature_csv(
            features=features,
            labels=labels,
            train_ids=data["train_ids"],
            val_ids=data["val_ids"],
            test_ids=data["test_ids"],
            output_path=FEATURE_OUTPUT_ROOT / f"features_split_{split_id}.csv"
        )

        split_data[split_id]["features"] = features
        split_data[split_id]["selected_shapelets"] = selected_shapelets

        valid = features["label"].dropna()
        feature_report.append({
            "split_id": split_id,
            "samples": len(features),
            "features": features.shape[1] - 3,
            "positive": int((valid == 1).sum()),
            "negative": int((valid == 0).sum())
        })

    # -------------------------------------------------------------------------------------------------------------------
    # 11. LightGBM model training
    # -------------------------------------------------------------------------------------------------------------------
    print("[11] LightGBM model training...")

    model_results, model_summary = run_lightgbm_training(
        feature_root=FEATURE_OUTPUT_ROOT,
        output_dir=MODEL_OUTPUT_ROOT,
        n_splits=5
    )

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

    print("\n[LightGBM]")
    for row in model_results.itertuples(index=False):
        print(
            f"Split {row.split_id} | "
            f"AUROC: {row.test_auroc:.4f} | "
            f"AUPRC: {row.test_auprc:.4f}"
        )

    auroc_row = model_summary.loc[
        model_summary["metric"] == "AUROC"
    ].iloc[0]

    auprc_row = model_summary.loc[
        model_summary["metric"] == "AUPRC"
    ].iloc[0]

    print(
        f"AUROC : {auroc_row['mean']:.4f} ± {auroc_row['sd']:.4f}"
    )
    print(
        f"AUPRC : {auprc_row['mean']:.4f} ± {auprc_row['sd']:.4f}"
    )


if __name__ == "__main__":
    main()