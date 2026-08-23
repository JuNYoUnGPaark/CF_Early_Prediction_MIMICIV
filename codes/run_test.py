from pathlib import Path
import pandas as pd

from step_1_3_filter_age import filter_age
from step_1_4_filter_mcs import filter_mcs
from step_1_5_filter_cf_data import filter_cf_data
from step_2_2_prepare_lactate import prepare_lactate
from step_2_3_prepare_height_weight import prepare_height_weight
from step_2_4_apply_permitted_ranges import apply_permitted_ranges
from step_2_5_remove_duplicates import remove_nonpharma_duplicates
from step_2_5_1_audit_pharma_duplicates import audit_pharma_duplicates
from step_3_1_classify_inputevents import classify_inputevents
from step_3_2_prepare_continuous_rate import prepare_continuous_rate
from step_3_3_build_candidates import build_acting_period_candidates
from step_3_3_prepare_noncontinuous_rate import \
    prepare_noncontinuous_effective_rate
from step_3_3_1_prepare_presence import prepare_noncontinuous_presence
from step_4_1_merge_nonpharma_concepts import merge_nonpharma_concepts
from step_4_1B_build_pharma_candidates import \
    build_pharma_target_candidates
from step_4_1_build_pharma_pool import build_pharma_pool
from step_4_2_merge_simultaneous_nonpharma import \
    merge_simultaneous_nonpharma
from step_4_2b_audit_pharma_simultaneous import audit_pharma_simultaneous

ROOT = Path(r"C://Users/sampa/OneDrive/문서/DAHS/3.1_subset")


# ============================================================
# 필요한 table load
# ============================================================
stays = pd.read_csv(ROOT / "icu" / "icustays.csv.gz")
admissions = pd.read_csv(ROOT / "hosp" / "admissions.csv.gz")
patients = pd.read_csv(ROOT / "hosp" / "patients.csv.gz")
d_items = pd.read_csv(ROOT / "icu" / "d_items.csv.gz")
chartevents = pd.read_csv(ROOT / "icu" / "chartevents.csv.gz", usecols=["stay_id", "itemid", "charttime", "valuenum"])
procedureevents = pd.read_csv(ROOT / "icu" / "procedureevents.csv.gz", usecols=["stay_id", "itemid"])
inputevents = pd.read_csv(ROOT / "icu" / "inputevents.csv.gz")
labevents = pd.read_csv(ROOT / "hosp" / "labevents.csv.gz", usecols=["hadm_id", "itemid", "charttime", "valuenum"])


# ============================================================
# 1-3. Age Exclusion
# ============================================================

stays, age_info, report_1_3 = filter_age(stays, admissions, patients)

# ============================================================
# 1-4. Mechanical Circulatory Support Exclusion
# ============================================================

stays, mcs_items, mcs_events, report_1_4 = filter_mcs(stays, d_items, chartevents, procedureevents)

# ============================================================
# 1-5. CF Data Availability
# ============================================================

stays, cf_data_info, report_1_5 = filter_cf_data(stays, chartevents)

# ============================================================
# 2-2. Blood Gas Artifact / Lactate Preparation
# ============================================================

lactate_events, report_2_2 = prepare_lactate(stays, chartevents, labevents)

# ============================================================
# 2-3. Height / Weight Artifact Preparation
# ============================================================

height_weight_events, report_2_3 = prepare_height_weight(stays, chartevents)

# ============================================================
# 2-4. Variable-specific Permitted Range
# ============================================================

chartevents, labevents, lactate_events, range_report, variable_report, report_2_4 = apply_permitted_ranges(
    stays, chartevents, labevents, lactate_events
)

# ============================================================
# 2-5. Raw Record Duplication Removal
# ============================================================

(
    chartevents,
    labevents,
    lactate_events,
    chart_duplicate_stats,
    lab_duplicate_stats,
    report_2_5_chart,
    report_2_5_lab
) = remove_nonpharma_duplicates(
    stays,
    chartevents,
    labevents
)

# # ============================================================
# # 2-5-1. Pharmaceutical Duplicate Audit
# # ============================================================

# (
#     pharma_duplicate_groups,
#     pharma_duplicate_rows,
#     pharma_duplicate_item_report,
#     pharma_category_report,
#     pharma_status_report,
#     report_2_5_1
# ) = audit_pharma_duplicates(
#     stays,
#     inputevents,
#     d_items
# )

# ============================================================
# 3-1. Pharmaceutical Administration Type
# ============================================================

(
    inputevents_classified,
    input_category_report,
    input_type_report,
    input_rate_report,
    input_amount_report,
    input_unknown_rows,
    report_3_1
) = classify_inputevents(
    stays,
    inputevents,
    d_items
)

# ============================================================
# 3-2. Continuous Rate Preparation
# ============================================================

continuous_events, continuous_rate_unresolved = \
    prepare_continuous_rate(inputevents_classified)

# ============================================================
# 3-3A. Build Acting-period Mapping Candidates
# ============================================================

acting_period_candidates, table4_drug_reference = \
    build_acting_period_candidates(
        inputevents_classified=inputevents_classified,
        table4_path=ROOT / "C://Users/sampa/OneDrive/문서/DAHS/datasets/Supplementary_TABLE4.xlsx",
        output_path=ROOT / "mimic4_noncontinuous_acting_period_candidates.csv"
    )

# ============================================================
# 3-3B. Non-continuous → Effective Continuous Rate
# ============================================================

noncontinuous_effective_events, report_3_3 = \
    prepare_noncontinuous_effective_rate(
        inputevents_classified,
        acting_period_candidates
    )

# ============================================================
# 3-3-1. Binary Presence Candidates
# ============================================================

noncontinuous_presence_events, report_3_3_1 = \
    prepare_noncontinuous_presence(
        inputevents_classified,
        acting_period_candidates
    )

# ============================================================
# 4-1A. Merge identical physiology / lab concepts
# ============================================================

(
    nonpharma_events,
    nonpharma_variable_report,
    nonpharma_mapping_report,
) = merge_nonpharma_concepts(
    stays,
    chartevents,
    labevents
)

# ============================================================
# 4-1B. Pharmaceutical Target Mapping Candidates
# ============================================================

pharma_target_candidates, pharma_input_items = \
    build_pharma_target_candidates(
        inputevents_classified=inputevents_classified,
        table4_path=ROOT / "C://Users/sampa/OneDrive/문서/DAHS/datasets/Supplementary_TABLE4.xlsx",
        output_path=ROOT / "mimic4_pharma_target_candidates.csv"
    )

# ============================================================
# 4-1C. Build final pharmaceutical pool
# ============================================================

(
    pharma_events,
    pharma_mapping,
    pharma_variable_report,
) = build_pharma_pool(
    continuous_events,
    noncontinuous_effective_events,
    pharma_target_candidates
)

# ============================================================
# 4-2A. Simultaneous physiology / lab → median
# ============================================================

nonpharma_merged = merge_simultaneous_nonpharma(
    nonpharma_events
)

# ============================================================
# 4-2B. Audit Simultaneous Pharmaceutical Events
# ============================================================

(
    pharma_simultaneous,
    pharma_mixed_unit,
    pharma_simultaneous_report,
    pharma_unit_report,
) = audit_pharma_simultaneous(
    pharma_events
)