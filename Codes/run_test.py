from pathlib import Path
import pandas as pd

from step_1_3_filter_age import filter_age
from step_1_4_filter_mcs import filter_mcs
from step_1_5_filter_cf_data import filter_cf_data

from step_2_2_prepare_lactate import prepare_lactate
from step_2_3_prepare_height_weight import prepare_height_weight
from step_2_4_apply_permitted_ranges import apply_permitted_ranges
from step_2_5_remove_duplicates import remove_nonpharma_duplicates

from step_3_1_classify_inputevents import classify_inputevents
from step_3_2_prepare_continuous_rate import prepare_continuous_rate
from step_3_3_build_candidates import build_acting_period_candidates
from step_3_3_prepare_noncontinuous_rate import prepare_noncontinuous_effective_rate
from step_3_3_1_prepare_presence import prepare_noncontinuous_presence

from step_4_1A_map_nonpharma import merge_nonpharma_concepts
from step_4_1B_map_pharma import map_pharma_concepts
from step_4_2A_merge_nonpharma_simultaneous import merge_nonpharma_simultaneous
from step_4_2B_merge_continuous_pharma import merge_continuous_pharma

from step_5_1_build_time_grid_60min import build_time_grid
from step_5_2_place_measurements_missingness_60min import place_measurements_on_grid
from step_5_3_project_pharma_to_grid import project_pharma_to_grid
from step_5_4A_calculate_imputation_parameters import calculate_imputation_parameters
from step_5_4B_adaptive_imputation import adaptive_impute_nonpharma
from step_5_5_prepare_static_variables import prepare_static_variables

from step_6_state_annotation_60min import annotate_circulatory_state

ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\3.1_subset")
DATASET_ROOT = Path(r"C:\Users\sampa\OneDrive\문서\DAHS\datasets")


# ============================================================
# 필요한 table load
# ============================================================

# ICU
stays = pd.read_csv(ROOT / "icu" / "icustays.csv.gz")
d_items = pd.read_csv(ROOT / "icu" / "d_items.csv.gz")
chartevents = pd.read_csv(
    ROOT / "icu" / "chartevents.csv.gz",
    usecols=["stay_id", "itemid", "charttime", "valuenum"]
)
procedureevents = pd.read_csv(
    ROOT / "icu" / "procedureevents.csv.gz",
    usecols=["stay_id", "itemid"]
)
inputevents = pd.read_csv(ROOT / "icu" / "inputevents.csv.gz")

# HOSP
admissions = pd.read_csv(ROOT / "hosp" / "admissions.csv.gz")
patients = pd.read_csv(ROOT / "hosp" / "patients.csv.gz")
d_labitems = pd.read_csv(
    ROOT / "hosp" / "d_labitems.csv.gz",
    usecols=["itemid", "label"]
)
labevents = pd.read_csv(
    ROOT / "hosp" / "labevents.csv.gz",
    usecols=["hadm_id", "itemid", "charttime", "valuenum"]
)
services = pd.read_csv(ROOT / "hosp" / "services.csv.gz")


# ============================================================
# 1. Cohort Filtering
# ============================================================

# 1-3. 입원 당시 나이를 계산해서 age 기준에 안 맞는 stay 제외
stays, age_info, report_1_3 = filter_age(
    stays,
    admissions,
    patients
)

# 1-4. ECMO / Impella 등 MCS가 있는 stay 제외
stays, mcs_items, mcs_events, report_1_4 = filter_mcs(
    stays,
    d_items,
    chartevents,
    procedureevents
)

# 1-5. 이후 grid를 만들 수 있도록 HR 기록이 없는 stay 제외
stays, cf_data_info, report_1_5 = filter_cf_data(
    stays,
    chartevents
)


# ============================================================
# 2. Raw Data Preprocessing
# ============================================================

# 2-2. chartevents + labevents의 Lactate를 현재 ICU stay 기준으로 모음
lactate_events, report_2_2 = prepare_lactate(
    stays,
    chartevents,
    labevents
)

# 2-3. Height / Weight를 가져오고 Weight 단위를 kg로 맞춤
height_weight_events, report_2_3 = prepare_height_weight(
    stays,
    chartevents
)

# 2-4. 논문에서 정한 permitted range 밖의 measurement 제거
(
    chartevents,
    labevents,
    lactate_events,
    range_report,
    variable_report,
    report_2_4
) = apply_permitted_ranges(
    stays,
    chartevents,
    labevents,
    lactate_events
)

# 2-5. 같은 시각에 중복된 raw numeric record를 논문 규칙대로 처리
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


# ============================================================
# 3. Pharmaceutical Preprocessing
# ============================================================

# 3-1. inputevents를 continuous / non-continuous 투여 형태로 구분
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

# 3-2. Continuous event에서 실제 사용할 투여 rate 준비
continuous_events, continuous_rate_unresolved = prepare_continuous_rate(
    inputevents_classified
)

# 3-3A. Non-continuous 약물의 acting period를 매칭하기 위한 후보 생성
acting_period_candidates, table4_drug_reference = build_acting_period_candidates(
    inputevents_classified=inputevents_classified,
    table4_path=DATASET_ROOT / "Supplementary_TABLE4.xlsx",
    output_path=ROOT / "mimic4_noncontinuous_acting_period_candidates.csv"
)

# 3-3B. Acting period가 매칭된 non-continuous event를 effective rate로 변환
noncontinuous_effective_events, report_3_3 = prepare_noncontinuous_effective_rate(
    inputevents_classified,
    acting_period_candidates
)

# 3-3-1. Non-continuous event를 투여 여부로 볼 수 있도록 presence 후보 생성
noncontinuous_presence_events, report_3_3_1 = prepare_noncontinuous_presence(
    inputevents_classified,
    acting_period_candidates
)


# ============================================================
# 4. Variable Mapping / Merging
# ============================================================

# 4-1A. 여러 raw ITEMID를 같은 non-pharma clinical variable로 통합
(
    nonpharma_events,
    nonpharma_variable_report,
    nonpharma_mapping_report,
    report_4_1A
) = merge_nonpharma_concepts(
    stays,
    chartevents,
    labevents
)

# 4-1B. 저자 mapping을 기준으로 사용할 pharmaceutical variable 매칭
(
    pharma_events,
    pharma_map,
    pharma_report
) = map_pharma_concepts(
    inputevents_classified=inputevents_classified,
    mimic_vars_path=DATASET_ROOT / "mimic_vars.csv",
    table4_path=DATASET_ROOT / "Supplementary_TABLE4.xlsx"
)

# 4-2A. 같은 시각의 같은 non-pharma variable을 하나로 합침
(
    nonpharma_merged,
    report_4_2A
) = merge_nonpharma_simultaneous(
    nonpharma_events
)

# 4-2B. 최종 continuous pharmaceutical event를 정리
(
    pharma_merged,
    pharma_merge_report,
    pharma_unit_report,
    report_4_2B
) = merge_continuous_pharma(
    continuous_events=continuous_events,
    pharma_map=pharma_map
)


# ============================================================
# 5. 1-hour Grid / Model Input Preparation
# ============================================================

# 5-1. 첫 HR부터 마지막 HR까지 1시간 간격의 time grid 생성
(
    time_grid,
    grid_stay_info,
    invalid_grid_stays,
    report_5_1
) = build_time_grid(
    stays=stays,
    nonpharma_merged=nonpharma_merged,
    grid_minutes=60,
    comparison_minutes=5,
    max_days=28
)

# 5-2. Raw measurement를 1시간 grid에 배치하고 missingness 확인
(
    raw_grid_values,
    missingness_report,
    report_5_2
) = place_measurements_on_grid(
    time_grid=time_grid,
    grid_stay_info=grid_stay_info,
    nonpharma_merged=nonpharma_merged,
    grid_minutes=60,
    comparison_minutes=5
)

# 5-3. Continuous drug interval을 1시간 grid의 rate로 변환
(
    pharma_grid,
    pharma_active_long,
    pharma_grid_report,
    report_5_3
) = project_pharma_to_grid(
    time_grid=time_grid,
    grid_stay_info=grid_stay_info,
    pharma_merged=pharma_merged,
    grid_minutes=60
)

# 5-4A. 변수별 sampling interval을 이용해 adaptive imputation parameter 계산
(
    imputation_params,
    sampling_intervals,
    report_5_4A
) = calculate_imputation_parameters(
    nonpharma_merged=nonpharma_merged,
    grid_stay_info=grid_stay_info
)

# 5-4B. 미래값은 보지 않고 non-pharma missing value를 adaptive imputation
(
    imputed_nonpharma_grid,
    imputation_report,
    report_5_4B
) = adaptive_impute_nonpharma(
    time_grid=time_grid,
    grid_stay_info=grid_stay_info,
    raw_grid_values=raw_grid_values,
    nonpharma_merged=nonpharma_merged,
    imputation_params=imputation_params,
    height_weight_events=height_weight_events,
    grid_minutes=60
)

# 5-5. Age / Sex / Height / Emergency / Surgical admission static variable 생성
(
    static_features,
    static_audit,
    report_5_5
) = prepare_static_variables(
    stays=stays,
    admissions=admissions,
    patients=patients,
    services=services,
    height_weight_events=height_weight_events
)


# ============================================================
# 6. State Annotation
# ============================================================

# Raw Lactate + raw MAP + raw drug interval로 현재 시점을 CF / NO_CF / AMBIGUOUS로 annotation
cf_state, report_6 = annotate_circulatory_state(
    time_grid=time_grid,
    raw_grid_values=raw_grid_values,
    nonpharma_merged=nonpharma_merged,
    pharma_merged=pharma_merged
)