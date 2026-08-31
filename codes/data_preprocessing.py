# -*- coding: utf-8 -*-
# name: data_preprocessing.py
# author: JunYoung Park
# date: 2026-08-26


import pandas as pd
import numpy as np
import re


# -----------------------------------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------------------------------
# (1) Permitted ranges 
CHARTEVENT_RANGES = {
    220045: ("Heart Rate", 0, 300),

    # Blood Pressure
    220050: ("ABP systolic", 10, 300),
    220051: ("ABP diastolic", 10, 175),
    220052: ("ABP mean", 10, 200),

    225309: ("ABP systolic", 10, 300),
    225310: ("ABP diastolic", 10, 175),
    225312: ("ABP mean", 10, 200),

    220179: ("NBP systolic", 10, 300),
    220180: ("NBP diastolic", 10, 175),
    220181: ("NBP mean", 10, 200),

    224842: ("Cardiac Output", 1, 20),
    227543: ("Cardiac Output", 1, 20),
    228178: ("Cardiac Output", 1, 20),
    228369: ("Cardiac Output", 1, 20),
    220088: ("Cardiac Output", 1, 20),

    220277: ("SpO2", 10, 100),
    228096: ("RASS", -5, 4),
    224695: ("Ventilator peak pressure", 5, 100),

    225668: ("Lactate", 0, 15),
    227467: ("INR", 0, 8),
    227444: ("C-reactive protein", 0, 600),
}

LABEVENT_RANGES = {
    50813: ("Lactate", 0, 15),
    51237: ("INR", 0, 8),
    50889: ("C-reactive protein", 0, 600),
}


# (2) Pharmaceutical variables
ADMINISTRATION_TYPE_MAP = {
    "Continuous Med": "continuous",
    "Continuous IV": "continuous",
    "Drug Push": "non-continuous",
    "Bolus": "non-continuous",
    "Non Iv Meds": "non-continuous",
}

# (3) Non-pharmaceutical variable mapping
CHARTEVENT_MAP = {
    220045: "Heart Rate",

    # Blood Pressure
    220050: "ABP systolic",
    220051: "ABP diastolic",
    220052: "ABP mean",

    225309: "ABP systolic",
    225310: "ABP diastolic",
    225312: "ABP mean",

    220179: "NBP systolic",
    220180: "NBP diastolic",
    220181: "NBP mean",

    224842: "Cardiac Output",
    227543: "Cardiac Output",
    228178: "Cardiac Output",
    228369: "Cardiac Output",
    220088: "Cardiac Output",

    220277: "SpO2",
    228096: "RASS",
    224695: "Ventilator peak pressure",

    225668: "Lactate",
    227467: "INR",

    220621: "Blood Glucose",
    225664: "Blood Glucose",
    226537: "Blood Glucose",

    227444: "C-reactive protein",
}

LABEVENT_MAP = {
    50813: "Lactate",
    51237: "INR",

    50931: "Blood Glucose",
    50809: "Blood Glucose",

    50889: "C-reactive protein",
}


# -----------------------------------------------------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------------------------------------------------
# 약물 이름 비교를 위해 대소문자 / 공백 / 기호 제거해주는 함수 
def _normalize_name(x):
    if pd.isna(x): return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


# acting period를 hour 단위로 변환해주는 함수 
def _acting_period_to_hours(x):
    x = str(x).strip().lower()
    if x.endswith("m"): return float(x[:-1]) / 60
    if x.endswith("h"): return float(x[:-1])
    if x.endswith("d"): return float(x[:-1]) * 24
    raise ValueError(f"Unknown acting period: {x}")


# -----------------------------------------------------------------------------------------------------------------------
# Age filtering
# -----------------------------------------------------------------------------------------------------------------------
def filter_age(stays: pd.DataFrame, admissions: pd.DataFrame, patients: pd.DataFrame):
    """
        stays -> ICU stays: subejct_id, stay_id, hadm_id
        admissions -> admissions table: subejct_id, hadm_id, admittime(입원 시각)
        patients -> patients table: subject_id, anchor_age, anchor_year
        
        Admission 당시 Age = anchor_age + (admission_year - anchor_age)
    """
    
    # 1. stays + admissions의 admittime + patients의 anchor_year, anchor_age 
    age_info = stays[["stay_id", "subject_id", "hadm_id"]].copy()
    
    age_info = age_info.merge(admissions[["subject_id", "hadm_id", "admittime"]],
                              on=["subject_id", "hadm_id"],  # on = 무엇을 기준으로 붙일지, key가 됨
                              how="left",  # how = 어느 행을 보존할지(여기서는 age_info가 left, admissions가 right)
                              validate="many_to_one")  # validate = 검사(left key 중복 가능, right(admissions)는 중복 X)
    
    age_info = age_info.merge(patients[["subject_id", "anchor_age", "anchor_year"]],
                              on="subject_id",
                              how="left",
                              validate="many_to_one")  
    
    # 2. admittime을 datetime으로 변환
    age_info["admittime"] = pd.to_datetime(age_info["admittime"], errors="coerce")
    
    # 3. Admission 당시 Age 계산
    age_info["admission_age"] = (age_info["anchor_age"] + (age_info["admittime"].dt.year - age_info["anchor_year"]))
    
    # 4. 제외 조건 
    # 4-1. anchor_age = 91인 경우 (89세 초과 환자는 anchor_age = 91로 topcoded돼있음)
    topcoded_age = age_info["anchor_age"].eq(91)
    
    # 4-2. Admisison Age가 missing인 경우
    missing_age = age_info["admission_age"].isna()
    
    # 4-3. Admisison Age가 16세 미만인 경우
    under_16 = (age_info["admission_age"] < 16)
    
    # 4-4. Admisison Age가 100세 초과인 경우 
    over_100 = (age_info["admission_age"] > 100)
    
    # 5. 제외하기 
    exclude = (topcoded_age | missing_age | under_16 | over_100)
    included_stay_ids = age_info.loc[~exclude, "stay_id"]
    filtered_stays = (stays[stays["stay_id"].isin(included_stay_ids)].copy().reset_index(drop=True))
    
    return filtered_stays


# -----------------------------------------------------------------------------------------------------------------------
# MCS filtering
# -----------------------------------------------------------------------------------------------------------------------
def filter_mcs(stays: pd.DataFrame, d_items: pd.DataFrame, chartevents: pd.DataFrame, procedureevents: pd.DataFrame):
    """
        stays -> 현재까지 필터링 통과한 ICU stay table
        d_items -> ITEMID에 대한 label, 이름 확보
        chartevents, procedureevents -> MCS 관련 variable, 기록 확인
    """
    
    # 현재까지 필터링 통과된 ICU stay 가져오기(stay_id 목록을 중복없이 모으기)
    current_stay_ids = set(stays["stay_id"])  
    
    # 1. MCS(Mechanical Circulatory Servies) keywords
    # HiRID: preprocessing/resource/varref_excel_v6.tsv에서 MCS 관련 variable 확인 (ECMO)
    # MIMIC-III: external_validation/mimic_vars.csv에서 External로 표현된 variable 확인
    # (ECMO, AssistDevice, Lef Ventricular Assist Device Flow, ...)
    pattern = (r"ECMO|" r"Impella|" r"LVAD|" r"RVAD|" 
               r"Assist Device|" r"Ventricular Assist Device|"
               r"Assit Device|" r"Ventricular Assit Device")
    
    # 2. d_items.label에서 MCS 관련 ITEMID 찾고 5가지 device family 중 하나로 분류
    mcs_items = d_items.loc[d_items["label"].fillna("").str.contains(pattern, case=False, regex=True)
                            & d_items["linksto"].isin(["chartevents", "procedureevents"]),
                            ["itemid", "label", "linksto"]].copy()
    mcs_items = mcs_items.drop_duplicates(subset="itemid").reset_index(drop=True)
    
    # 3. chartevents에서 5개 MCS ITEMID가 있는 stay_id, itemid 가져오기 
    # d_items에서 찾아낸 MCS 관련 ITEMID 중, chartevents에 존재하는 ITEMID 집합만들기 
    chart_itemids = set(mcs_items.loc[mcs_items["linksto"] == "chartevents", "itemid"])
    # 현재 분석대상인 stay_id면서 itemid가 MCS 관련된 itemid인 행만 남기기   
    chart_mcs = chartevents.loc[chartevents["stay_id"].isin(current_stay_ids)
                                & chartevents["itemid"].isin(chart_itemids),
                                ["stay_id", "itemid"]].copy()
    # chartevents에서 가져온 기록임을 표시하기 위해 source_table 컬럼 추가 
    chart_mcs["source_table"] = "chartevents"
    
    # 4. procedureevents에서 5개 MCS ITEMID가 있는 stay_id, itemid 가져오기 
    procedure_itemids = set(mcs_items.loc[mcs_items["linksto"] == "procedureevents", "itemid"])
    procedure_mcs = procedureevents.loc[procedureevents["stay_id"].isin(current_stay_ids)
                                        & procedureevents["itemid"].isin(procedure_itemids),
                                        ["stay_id", "itemid"]].copy()
    procedure_mcs["source_table"] = "procedureevents"
    
    # 5. 두 table MCS 기록 합치기 
    mcs_events = pd.concat([chart_mcs, procedure_mcs], ignore_index=True)
    
    # 6. MCS가 실제 기록된 ICU stay 찾고 그 stay는 제외하기
    mcs_stay_ids = set(mcs_events["stay_id"].dropna().unique())
    filtered_stays = stays.loc[~stays["stay_id"].isin(mcs_stay_ids)].copy().reset_index(drop=True)
    
    return filtered_stays


# -----------------------------------------------------------------------------------------------------------------------
# CF availability filtering
# -----------------------------------------------------------------------------------------------------------------------
def filter_cf_data(stays: pd.DataFrame, chartevents: pd.DataFrame):
    """
        stays -> 현재까지 필터링 통과한 ICU stay table
        chartevents -> HR variable, HR이 missing인 stay 제외
    """
    
    # 현재까지 필터링 통과된 ICU stay 가져오기(stay_id 목록을 중복없이 모으기)
    current_stay_ids = set(stays["stay_id"])
    
    # 실제 HR 측정값 찾기 
    hr = chartevents.loc[chartevents["stay_id"].isin(current_stay_ids)
                         & (chartevents["itemid"] == 220045)
                         & chartevents["valuenum"].notna(),
                         ["stay_id", "itemid", "valuenum"]].copy()
    
    # HR 기록이 없는 ICU stay 제외 
    hr_stay_ids = set(hr["stay_id"].unique())
    no_hr_stay_ids = current_stay_ids - hr_stay_ids
    filtered_stays = stays.loc[~stays["stay_id"].isin(no_hr_stay_ids)].copy().reset_index(drop=True)
    
    return filtered_stays


# -----------------------------------------------------------------------------------------------------------------------
# Apply permitted range
# -----------------------------------------------------------------------------------------------------------------------
def restrict_to_current_cohort(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame,):
    """
        stays -> 현재까지 필터링 통과한 ICU stays
        chartevents -> 현재까지 통과된 stay만 남기기
        labevents -> 현재까지 통과된 stay만 남기기
    """
    
    # 1. chartevents에서 현재까지 필터링 통과한 stay만 가져오기 
    current_stay_ids = set(stays["stay_id"])
    chartevents = chartevents.loc[chartevents["stay_id"].isin(current_stay_ids)].copy().reset_index(drop=True)
    
    # 2. labevent에는 stay_id가 없는 관계로 
    # 같은 hadm_id && ICU intime <= charttime <= ICU outtime에 해당하는 row만 가져오기
    windows = stays[["hadm_id", "intime", "outtime"]].copy()
    
    # intime, outtime, charttime을 Datetime으로 변환하기
    windows["intime"] = pd.to_datetime(windows["intime"], errors="coerce")
    windows["outtime"] = pd.to_datetime(windows["outtime"], errors="coerce")
    labevents = labevents.copy()
    labevents["charttime"] = pd.to_datetime(labevents["charttime"], errors="coerce")
    
    # 각 row에 임시 번호 붙이기 -> hadm_id하나에 여러 stay들이 붙게됨
    current_hadm_ids = set(stays["hadm_id"].dropna())
    labevents = labevents.loc[labevents["hadm_id"].isin(current_hadm_ids)].copy()
    labevents["_row_id"] = range(len(labevents))
    
    # 현재 cohort의 ICU stay 중 하나 이상의 time window에 포함되는 lab row만 남김
    matched = labevents.merge(windows, on="hadm_id", how="inner")
    matched = matched[(matched["charttime"] >= matched["intime"])
                      & (matched["charttime"] <= matched["outtime"])]
    
    # 통과한 row 번호만 저장
    valid_ids = set(matched["_row_id"])    
    
    # 원래 labevents에 통과한 row들만 남기기
    labevents = (labevents[labevents["_row_id"].isin(valid_ids)].drop(columns="_row_id").reset_index(drop=True))
    
    return chartevents, labevents


def filter_permitted_ranges(df: pd.DataFrame, rules: dict):
    """
        df -> 현재까지 통과된 stay만 보관중인 chartevents & labevents
        rules -> itemid별 permitted range
    """
    
    # 1. 처음에는 모든 row를 False로 만들어두기 
    filtered = df.copy()
    remove_mask = pd.Series(False, index=filtered.index)  
    
    for itemid, (variable, lower, upper) in rules.items():
        """
            2. permitted range(rules)를 하나씩 꺼내서 
                itemid = 220045
                variable = "Heart Rate"
                lower = 0
                upper = 300
            로 가져와서 범위에 밖인 row만 True로 만든다. 
        """
        
        target_mask = ((filtered["itemid"] == itemid) & filtered["valuenum"].notna())
        out_of_range_mask = (target_mask & ~filtered["valuenum"].between(lower, upper, inclusive="both"))
        remove_mask = remove_mask | out_of_range_mask  # 반복문을 돌면서 remove_mask에 변수별로 mask가 쌓인다. 
    
    # 3. 삭제 대상이 아닌 row만 삭제하기. (mask=True인 것만)
    filtered = (filtered.loc[~remove_mask].copy().reset_index(drop=True))
    
    return filtered


# -----------------------------------------------------------------------------------------------------------------------
# Duplicates processing
# -----------------------------------------------------------------------------------------------------------------------
def remove_numeric_duplicates(df: pd.DataFrame, id_col: str):
    """
        df -> 현재까지 통과된 stay만 보관중인 chartevents & labevents
        id_col -> 어떤 단위로 같은 환자 기록인지 판단할지 정함
    """
    
    work = df.copy()
    # charttime을 Datetime으로 변환하기 
    work["charttime"] = pd.to_datetime(work["charttime"], errors="coerce")
    
    # 1. duplicate 처리 가능한 row / 아닌 row 분리 
    # 숫자값과 시간이 둘 다 있는 row만 True
    target_mask = (work["valuenum"].notna() & work["charttime"].notna())
    numeric = work.loc[target_mask].copy()  # 처리 가능한 row
    untouched = work.loc[~target_mask].copy()  # 아닌 row
    
    # 2. 원래의 row 순서를 남기기 (어떤걸 남길지 등 구분을 하기 위함)
    numeric["_row_order"] = np.arange(len(numeric))
    
    # 3. itemid별 Global SD 계산하기    
    global_std = numeric.groupby("itemid")["valuenum"].std()
    
    # 4. duplicate인 row의 상태를 요약한 table 만들기 
    # stay_id, itemid, charttime 모두 같으면 duplicate 후보로 
    # agg: 각 그룹을 여러 통계값으로 요약 
    # n > 1, n_unique = 1이면 같은 값이 중복된 case
    # n > 1, n_unique > 1이면 같은 시간인데, 서로 다른 값으로 기록됨 -> 얼마나 차이나는지 추가 판단 필요 
    stats = (numeric.groupby([id_col, "itemid", "charttime"])["valuenum"].agg(n="size",  # 해당 그룹의 row의 개수
                                                                              n_unique="nunique",  # 해당 그룹의 nunique
                                                                              mean="mean",  # 해당 그룹의 mean
                                                                              std="std").reset_index())  # 해당 그룹의 std
    
    # 5. 실제 duplicate만 남기기 
    stats = stats.loc[stats["n"] > 1].copy() 
    
    # 6. 실제 duplicate만 남긴 table에서 itemid별로 Global SD 붙이기 
    stats["global_std"] = stats["itemid"].map(global_std)
    
    # 7. Case별로 처리하기 
    stats["action"] = "remove"  # 기본 상태 = 삭제로 설정 
    
    # Case1: 값이 전부 같으면 하나만 남기기 
    stats.loc[stats["n_unique"] == 1, "action"] = "keep_one"
    
    # Case2: 값이 다를땐 얼마나 다른지 검사하고 조금만 다르면 평균으로 대체하기 
    small_diff = ((stats["n_unique"] > 1)  # 서로 다른 값이여야하고
                  & stats["global_std"].notna()  # Global SD가 계산되어 있어야하고
                  & (stats["global_std"] > 0)  # Global SD가 0보다 커야하고 
                  & (stats["std"] < 0.05 * stats["global_std"]))  # duplicated끼리의 SD가 Gloabl SD 5%보다 작으면
    stats.loc[small_diff, "action"] = "mean"  # 평균으로 
    
    # 각 원본 row에 처리 방법 붙이기 
    marked = numeric.merge(stats[[id_col, "itemid", "charttime"] + ["action", "mean"]],
                           on=[id_col, "itemid", "charttime"],
                           how="left")
    
    # duplicate가 아닌 row(정상 row는 처리 방법이 NaN임)
    normal = marked.loc[marked["action"].isna()].copy()
    
    # 완전히 같은 duplicate -> 첫번째 _row_order값 하나만 남김
    keep_one = (marked.loc[marked["action"] == "keep_one"]
                .sort_values("_row_order")
                .drop_duplicates([id_col, "itemid", "charttime"], keep="first")
                .copy())

    # 작은 차이 duplicate -> mean값
    averaged = (marked.loc[marked["action"] == "mean"]
                .sort_values("_row_order")
                .drop_duplicates([id_col, "itemid", "charttime"], keep="first")
                .copy())  # 이렇게 일단 1개만 남기고 
    # 남겨진 값들을 평균값으로 바꾸기 
    averaged["valuenum"] = averaged["mean"]
    
    # action == "remove"인 (duplicated끼리의 SD가 Global SD 5%보다 큰) row는 애초에 concat을 안해서 자동 삭제됨 
    cleaned_numeric = pd.concat([normal, keep_one, averaged], ignore_index=True)
    
    # 8. 임시로 만들었던 column들 삭제 
    cleaned_numeric = cleaned_numeric.drop(columns=["_row_order", "action", "mean"], errors="ignore")
    
    # 9. 처음에 duplicate 처리 대상이 아니었던 row 다시 붙이기 
    cleaned = pd.concat([untouched, cleaned_numeric], ignore_index=True)
    
    # 10. ID->시간->itemid순으로 정렬
    cleaned = cleaned.sort_values([id_col, "charttime", "itemid"]).reset_index(drop=True)
    
    return cleaned


# -----------------------------------------------------------------------------------------------------------------------
# 약물 변수 Continuous / Non-continuous 분류
# -----------------------------------------------------------------------------------------------------------------------
def classify_inputevents(stays: pd.DataFrame, inputevents: pd.DataFrame, d_items: pd.DataFrame):
    """
        stays -> 현재까지 통과/처리된 ICU stays
        inputevents -> 약물 정보 
        d_items -> 약물 label mapping 
    """
    
    # 1. 현재까지 filtering, processing 통과한 ICU stay만 가져오기
    current_stays = set(stays["stay_id"])
    
    # inputevents
    events = inputevents.loc[inputevents["stay_id"].isin(current_stays)].copy()
    
    # inputevents의 starttime, endtime Datetime으로 변환하기 
    events["starttime"] = pd.to_datetime(events["starttime"], errors="coerce")
    events["endtime"] = pd.to_datetime(events["endtime"], errors="coerce")
    
    # 2. itemid의 약물 이름 붙이기 
    item_info = (d_items[["itemid", "label"]].drop_duplicates("itemid"))
    events = events.merge(item_info, on="itemid", how="left")
    
    # 3. 투여 형태 분류
    events["administration_type"] = events["ordercategorydescription"].map(ADMINISTRATION_TYPE_MAP).fillna("unknown")
    
    return events


# -----------------------------------------------------------------------------------------------------------------------
# Continuous pharmaceutical -> rate representation
# -----------------------------------------------------------------------------------------------------------------------
def prepare_continuous_rate(events):
    """
        events -> inputevents중 Continuous/Non-continuous 분류가 된 events
    """
    
    # 1. 투여 형태가 continuous인 약물 event들만 가져오기 
    continuous = events.loc[events["administration_type"] == "continuous"].copy()
    
    # 2. 이미 기록되어있는 rate가 있으면 그대로 사용 
    continuous["continuous_rate"] = continuous["rate"]
    continuous["continuous_rate_uom"] = continuous["rateuom"]
    continuous["rate_source"] = np.where(continuous["rate"].notna(), "recorded", "missing")
    
    # 3. rate가 없으면 amount / duration으로 직접 계산
    # 실제 투여 소요시간(duration) 계산하기 
    continuous["duration_hours"] = (continuous["endtime"] - continuous["starttime"]).dt.total_seconds() / 3600
    reconstruct = (continuous["rate"].isna() & continuous["amount"].notna() & continuous["duration_hours"] > 0)
    
    continuous.loc[reconstruct, "continuous_rate"] = \
    (continuous.loc[reconstruct, "amount"] / continuous.loc[reconstruct, "duration_hours"])
    continuous.loc[reconstruct, "continuous_rate_uom"] = (continuous.loc[reconstruct, "amountuom"].astype(str) + "/hour")
    continuous.loc[reconstruct, "rate_source"] = "derived_amount_duration"
    
    return continuous


# -----------------------------------------------------------------------------------------------------------------------
# Non-continuous + acting period -> effective rate representation
# -----------------------------------------------------------------------------------------------------------------------
def match_acting_period(events: pd.DataFrame, table4_path: str):
    """
        # Supplementary Table IV와 Non-continuous pharmaceutical 비교 -> mapping 만들기 
        events -> inputevents중 Continuous/Non-continuous 분류가 된 events
        table4_path -> Supplementary Table IV 위치
    """
    
    # 1. 현재 non-continuous itemid와 label 
    items = (events.loc[events["administration_type"] == "non-continuous", ["itemid", "label"]].drop_duplicates().copy())
    
    # 2. label 정규화하기
    items["name_norm"] = items["label"].apply(_normalize_name)
    
    # 3. Supplementary Table4 불러오기
    table4 = pd.read_excel(table4_path, sheet_name="drugs")
    table4.columns = [str(c).strip() for c in table4.columns]
    
    # 4. acting period가 정의된 약물들만 MIMIC IV와 매칭되면 mapping해주기
    ref = table4[["constituent drugs (if relevant)", "acting period (individual)"]].copy()
    ref = ref.dropna(subset=["constituent drugs (if relevant)", "acting period (individual)"])
    ref = ref.rename(columns={"constituent drugs (if relevant)": "table4_drug", "acting period (individual)": "acting_period"})
    ref["name_norm"] = (ref["table4_drug"].apply(_normalize_name))
    
    # exact matching -> MIMIC label & Table 4 drug name
    mapping = items.merge(ref[["name_norm", "table4_drug", "acting_period"]], on="name_norm", how="left")
    mapping = (mapping.loc[mapping["table4_drug"].notna()].drop_duplicates("itemid").copy())
    mapping["acting_period_hours"] = (mapping["acting_period"].apply(_acting_period_to_hours))
    mapping = mapping[["itemid", "table4_drug", "acting_period", "acting_period_hours"]]
    
    return mapping


def prepare_noncontinuous_rate(events: pd.DataFrame, mapping: pd.DataFrame):
    """
        events -> 전체 약물관련 events
        mapping -> _match_acting_period 함수를 통해 만든 mapping 
    """
    
    # 1. non-continuous 약물인 events 가져오기 
    noncontinuous = events.loc[events["administration_type"] == "non-continuous"].copy()
    
    # 2. Table 4 acting period가 매칭된 event에 mapping 합치기
    effective = noncontinuous.merge(mapping, on="itemid", how="inner")
    
    # 3. Effective rate 계산하기 (amount / acting period)
    effective["effective_rate"] = (effective["amount"] / effective["acting_period_hours"])
    effective["effective_rate_uom"] = (effective["amountuom"].astype(str) + "/hour")
    
    # 4. Acting period 동안 Effective rate만큼 약효가 지속된다고 표현 
    effective["effective_starttime"] = effective["starttime"]
    effective["effective_endtime"] = (effective["effective_starttime"] 
                                      + pd.to_timedelta(effective["acting_period_hours"], unit="h"))
    
    return effective 


# -----------------------------------------------------------------------------------------------------------------------
# 아직도 mapping이 안된 Non-continuous pharmaceutical -> binary presence 
# -----------------------------------------------------------------------------------------------------------------------
def prepare_noncontinuous_presence(events: pd.DataFrame, mapping: pd.DataFrame):
    # 1. non-continuous 약물인 events 가져오기 
    noncontinuous = events.loc[events["administration_type"] == "non-continuous"].copy()
    
    # 2. 이미 effective rate으로 표현된 itemid는 제외하기 
    quantitative_itemids = set(mapping["itemid"])
    presence = noncontinuous.loc[~noncontinuous["itemid"].isin(quantitative_itemids)].copy()
    
    # 3. event의 starttime~endtime 동안 presence=1로 설정 
    presence["presence"] = 1
    presence["presence_starttime"] = presence["starttime"]
    presence["presence_endtime"] = presence["endtime"]
    
    return presence


# -----------------------------------------------------------------------------------------------------------------------
# Non-pharma variable merge & merge시 발생한 duplicate -> median
# -----------------------------------------------------------------------------------------------------------------------
def map_nonpharma_concepts(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame):
    # 1. chartevents
    chart = chartevents.loc[chartevents["itemid"].isin(CHARTEVENT_MAP)
                            & chartevents["valuenum"].notna(),
                            ["stay_id", "itemid", "charttime", "valuenum"]].copy()
    chart["charttime"] = pd.to_datetime(chart["charttime"], errors="coerce")
    chart = chart.dropna(subset=["charttime"])
    
    # CHARTEVENT_MAP을 이용해서 itemid mapping시키기
    chart["variable"] = chart["itemid"].map(CHARTEVENT_MAP)
    
    # Glucose 단위 변경: mg/dL -> mmol/L
    glucose = chart["itemid"].isin([220621, 225664, 226537])
    chart.loc[glucose, "valuenum"] /= 18.0
    
    # 원래 어떤 itemid였는지 기록하기 위해서 
    chart["source_itemid"] = chart["itemid"]
    
    chart["source"] = "chartevents"
    chart = chart[["stay_id", "charttime", "variable", "valuenum", "source_itemid", "source"]]
    
    # 2. labevents
    lab = labevents.loc[labevents["itemid"].isin(LABEVENT_MAP)
                        & labevents["valuenum"].notna(),
                        ["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
    lab = lab.dropna(subset=["charttime"])
    
    # lab row 순서대로 번호 매겨놓기.
    lab["_row_id"] = range(len(lab))
    
    # ICU stay 불러오기 
    windows = stays[["stay_id", "hadm_id", "intime", "outtime"]].copy()
    windows["intime"] = pd.to_datetime(windows["intime"], errors="coerce")
    windows["outtime"] = pd.to_datetime(windows["outtime"], errors="coerce")
    
    # stay를 lab에 merge
    lab = lab.merge(windows, on="hadm_id", how="inner")
    
    # ICU stay 시간 안에 해당되는 lab만 골라내기 
    lab = lab.loc[(lab["charttime"] >= lab["intime"]) & (lab["charttime"] <= lab["outtime"])].copy()
    
    # 하나의 lab row가 둘 이상의 stay에 들어가면 제외 
    stay_count = lab.groupby("_row_id")["stay_id"].nunique()
    ambiguous_ids = set(stay_count[stay_count > 1].index)
    lab = lab.loc[~lab["_row_id"].isin(ambiguous_ids)].copy()
    
    # CHARTEVENT_MAP을 이용해서 itemid mapping시키기
    lab["variable"] = lab["itemid"].map(LABEVENT_MAP)
    
    # Glucose 단위 변경: mg/dL -> mmol/L
    glucose = lab["itemid"].isin([50931, 50809])
    lab.loc[glucose, "valuenum"] /= 18.0
    
    # 원래 어떤 itemid였는지 기록하기 위해서 
    lab["source_itemid"] = lab["itemid"]
    
    lab["source"] = "labevents"
    lab = lab[["stay_id", "charttime", "variable", "valuenum", "source_itemid", "source"]]
    
    # 3. chartevents + labevents
    events = pd.concat([chart, lab], ignore_index=True)
    events = (events.sort_values(["stay_id", "charttime", "variable"]).reset_index(drop=True))
    
    return events


def merge_nonpharma_simultaneous(events):
    # 1. Lactate: 같은 시각에 chart/lab이 같이 있으면 chartevents 우선
    lactate = events.loc[events["variable"] == "Lactate"].copy()
    lactate["priority"] = lactate["source"].map({"chartevents": 0, "labevents": 1})

    lactate = (lactate.sort_values(["stay_id", "charttime", "priority"])
               .drop_duplicates(["stay_id", "charttime", "variable"], keep="first")
               [["stay_id", "charttime", "variable", "valuenum"]])

    # 2. 나머지 변수: 기존과 동일하게 같은 시각 값 median
    others = events.loc[events["variable"] != "Lactate"].copy()
    others = (others.groupby(["stay_id", "charttime", "variable"], as_index=False)["valuenum"].median())

    # 3. 다시 합치기
    merged = pd.concat([lactate, others], ignore_index=True)
    merged = merged.sort_values(["stay_id", "charttime", "variable"]).reset_index(drop=True)
    
    return merged


# -----------------------------------------------------------------------------------------------------------------------
# MAP supplementation using SBP / DBP / NBP
# -----------------------------------------------------------------------------------------------------------------------
def supplement_map(merged: pd.DataFrame):
    """
        MAP 우선순위
        1. 직접 측정된 invasive ABP mean
        2. invasive ABP systolic + diastolic로 계산
        3. 직접 측정된 NBP mean
        4. NBP systolic + diastolic로 계산

        MAP = (SBP + 2 * DBP) / 3
    """

    df = merged.copy()

    # 1. BP 변수만 wide format으로 변환
    bp_variables = [
        "ABP systolic", "ABP diastolic", "ABP mean",
        "NBP systolic", "NBP diastolic", "NBP mean"
    ]

    bp = df.loc[df["variable"].isin(bp_variables)].copy()

    wide = (
        bp.pivot_table(
            index=["stay_id", "charttime"],
            columns="variable",
            values="valuenum",
            aggfunc="median"
        )
        .reset_index()
    )

    # 없는 column이 있더라도 계산 가능하도록 생성
    for col in bp_variables:
        if col not in wide.columns:
            wide[col] = np.nan

    # 2. invasive SBP / DBP로 MAP 계산
    wide["calculated_abp_mean"] = (
        wide["ABP systolic"] + 2 * wide["ABP diastolic"]
    ) / 3

    # 3. NBP SBP / DBP로 MAP 계산
    wide["calculated_nbp_mean"] = (
        wide["NBP systolic"] + 2 * wide["NBP diastolic"]
    ) / 3

    # 4. MAP 우선순위 적용
    wide["final_map"] = (
        wide["ABP mean"]
        .fillna(wide["calculated_abp_mean"])
        .fillna(wide["NBP mean"])
        .fillna(wide["calculated_nbp_mean"])
    )

    map_rows = wide.loc[
        wide["final_map"].notna(),
        ["stay_id", "charttime", "final_map"]
    ].copy()

    map_rows["variable"] = "ABP mean"
    map_rows = map_rows.rename(columns={"final_map": "valuenum"})

    # 5. 기존 ABP mean은 제거하고 보완된 MAP로 교체
    df = df.loc[df["variable"] != "ABP mean"].copy()

    df = pd.concat(
        [
            df,
            map_rows[["stay_id", "charttime", "variable", "valuenum"]]
        ],
        ignore_index=True
    )

    # NBP는 MAP 보완용으로만 사용하므로 최종 feature에서는 제거
    df = df.loc[
        ~df["variable"].isin([
            "NBP systolic",
            "NBP diastolic",
            "NBP mean"
        ])
    ].copy()

    df = (
        df.sort_values(["stay_id", "charttime", "variable"])
        .reset_index(drop=True)
    )

    return df

    
# -----------------------------------------------------------------------------------------------------------------------
# Pharma variable merge & merge시 발생한 duplicate 처리
# -----------------------------------------------------------------------------------------------------------------------
def map_pharma_concepts(events: pd.DataFrame, mimic_vars_path: str, table4_path: str):
    items = events[["itemid", "label"]].drop_duplicates().copy()
    items["name_norm"] = items["label"].apply(_normalize_name)
    
    # 1. mimic_vars.csv mapping
    mimic = pd.read_csv(mimic_vars_path)
    include = mimic["include"].astype(str).str.lower().eq("true")
    mid = mimic["mID"].fillna("").astype(str)
    
    pharma = (mid.str.lower().str.startswith("pm") | mid.str.lower().str.startswith("m_pm"))
    input_table = (mimic["table"].fillna("").astype(str).str.lower().str.contains("inputevents"))
    
    direct = mimic.loc[include & pharma & input_table].copy()
    direct["itemid"] = pd.to_numeric(direct["ITEM_ID"], errors="coerce")
    direct = direct.dropna(subset=["itemid"])
    direct["itemid"] = direct["itemid"].astype(int)
    direct["pharma_variable"] = (direct["varname2"].fillna(direct["varname (mimic?)"]))
    direct = (direct[["itemid", "mID", "pharma_variable"]].rename(columns={"mID": "pharma_id"}).drop_duplicates("itemid"))
    
    # 2. Supplementary TABLE IV mapping
    master = pd.read_excel(table4_path, sheet_name="master list")
    eligible_ids = set()
    
    for _, row in master.iterrows():
        pharma_id = str(row["ID"]).strip()
        mimic_ids = row["MIMIC varids"]
        
        if (pharma_id.lower().startswith("pm")
            and pd.notna(mimic_ids)
            and re.search(r"\d+", str(mimic_ids))
        ):
            eligible_ids.add(pharma_id)

    drugs = pd.read_excel(table4_path, sheet_name="drugs")
    drugs["temp: ID"] = drugs["temp: ID"].ffill()
    drugs["temp: bern name"] = drugs["temp: bern name"].ffill()
    
    refs = []
    
    for col in ["drug", "constituent drugs (if relevant)"]:
        ref = drugs[["temp: ID", "temp: bern name", col]].dropna(subset=[col]).copy()
        ref = ref.loc[ref["temp: ID"].astype(str).isin(eligible_ids)]
        ref["name_norm"] = ref[col].apply(_normalize_name)
        ref = ref.rename(columns={"temp: ID": "pharma_id", "temp: bern name": "pharma_variable"})
        refs.append(ref[["name_norm", "pharma_id", "pharma_variable"]])
    
    # refs라는 List에 있는 여러 DataFrame을 세로로 이어붙이는 것 
    table4_ref = pd.concat(refs, ignore_index=True).drop_duplicates()
    table4_match = items.merge(table4_ref, on="name_norm", how="inner")
    
    # 한 itemid가 여러 pharma에 매칭되면 사용하지 않음
    counts = table4_match.groupby("itemid")["pharma_id"].nunique()
    safe_ids = set(counts[counts == 1].index)
    table4_match = (table4_match.loc[table4_match["itemid"].isin(safe_ids), 
                                     ["itemid", "pharma_id", "pharma_variable"]].drop_duplicates("itemid"))
    
    # mimic_vars.csv를 우선으로 사용하고, 이미 mapping되는 itemid는 TABLE IV mapping에서 제외
    direct_ids = set(direct['itemid'])
    table4_match = table4_match.loc[~table4_match["itemid"].isin(direct_ids)]
    
    # direct mapping + Table 4에서 추가로 매핑된 ITEMID를 합쳐 최종 pharma mapping 생성
    pharma_map = (pd.concat([direct, table4_match], ignore_index=True).drop_duplicates("itemid"))
    
    pharma_map = pharma_map.loc[pharma_map["itemid"].isin(items["itemid"])].copy()
    
    return pharma_map


def merge_mapped_pharma(continuous: pd.DataFrame, pharma_map: pd.DataFrame):
    # canonical pharma mapping
    df = continuous.merge(pharma_map[["itemid", "pharma_id", "pharma_variable"]], on="itemid", how="inner").copy()

    df = df.dropna(subset=["stay_id", "starttime", "endtime", "continuous_rate"])
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")  

    # 완전히 동일한 infusion duplicate 제거
    duplicate_cols = ["stay_id", "pharma_id", "starttime", "endtime", "continuous_rate", "continuous_rate_uom"]
    df = df.drop_duplicates(subset=duplicate_cols, keep="first")

    # end <= start인 잘못된 interval 제거
    df = df.loc[df["endtime"] > df["starttime"]].copy()
    
    df = (df.sort_values(["stay_id", "pharma_id", "starttime", "endtime"]).reset_index(drop=True))

    return df


# -----------------------------------------------------------------------------------------------------------------------
# prepare_height_weight
# -----------------------------------------------------------------------------------------------------------------------
def prepare_height_weight(stays: pd.DataFrame, chartevents: pd.DataFrame):
    current_stay_ids = set(stays["stay_id"])

    hw = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin([226730, 226512, 226531])
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "charttime", "valuenum"]
    ].copy()

    hw["charttime"] = pd.to_datetime(hw["charttime"], errors="coerce")
    hw = hw.dropna(subset=["charttime"])

    hw["variable"] = hw["itemid"].map({
        226730: "Height",
        226512: "Weight",
        226531: "Weight"
    })

    hw["value"] = hw["valuenum"]

    lbs = hw["itemid"] == 226531
    hw.loc[lbs, "value"] = hw.loc[lbs, "valuenum"] * 0.453592

    return hw[
        ["stay_id", "charttime", "variable", "value"]
    ].sort_values(
        ["stay_id", "charttime", "variable"]
    ).reset_index(drop=True)