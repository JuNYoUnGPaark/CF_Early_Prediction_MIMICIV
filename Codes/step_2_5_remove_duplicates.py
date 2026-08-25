import numpy as np
import pandas as pd

# ============================================================
# 2-5. Duplicated Variable Processing
# ============================================================
# 원 논문에서는 같은 patient + 같은 variable + 같은 timestamp에
# 여러 값이 기록된 경우 duplicate artifact로 보고 처리함.
#
# HiRID에서는 non-pharmaceutical variable에 대해:
#
#   1) duplicate 값이 모두 같음
#      → 하나만 유지
#
#   2) duplicate 값이 서로 다름
#      → duplicate들의 SD와 해당 variable 전체 SD를 비교
#
#      duplicate SD < global SD의 5%
#      → 서로 거의 비슷한 값이라고 보고 평균값 하나 유지
#
#      duplicate SD >= global SD의 5%
#      → 값 차이가 너무 크다고 보고 unreliable → 전부 삭제
#
# 하지만 MIMIC-III external validation에서는
# HiRID의 5%-SD duplicate algorithm을 그대로 적용하지 않고
# fixed permitted range 중심으로 artifact를 처리했음.
#
# MIMIC-IV에서도 HiRID-specific duplicate rule을 그대로 적용하기보다는
# MIMIC의 실제 record 구조를 기준으로 처리 방법을 Adapt할 필요가 있음.
#
# 현재 duplicate 확인 대상:
#
#   chartevents
#   → 현재 cohort의 stay_id에 해당하고 valuenum이 있는 numeric record
#
#   labevents
#   → 현재 2878 ICU time window로 제한된 뒤
#      valuenum이 있는 numeric record
#
#   inputevents
#   → pharmaceutical record이므로 여기서는 처리하지 않고
#      별도의 pharmaceutical processing에서 확인
#
# ============================================================

def _remove_numeric_duplicates(df: pd.DataFrame, id_col: str, table_name: str):

    # datetime으로 변환
    work = df.copy()
    work["charttime"] = pd.to_datetime(work["charttime"], errors="coerce")

    # numeric measurement와 나머지 row 분리
    target_mask = work["valuenum"].notna() & work["charttime"].notna()
    numeric = work.loc[target_mask].copy()
    untouched = work.loc[~target_mask].copy()
    numeric["_row_order"] = np.arange(len(numeric))
    before_rows = len(numeric)

    # raw ITEMID별 global SD 계산
    global_std = (
        numeric.groupby("itemid")["valuenum"]
        .std()
        .rename("global_std")
        .reset_index()
    )

    #  same patient/stay + ITEMID + timestamp 기준 group 생성
    group_cols = [id_col, "itemid", "charttime"]
    group_stats = (
        numeric.groupby(group_cols)
        .agg(
            n_rows=("valuenum", "size"),
            n_unique_values=("valuenum", "nunique"),
            duplicate_mean=("valuenum", "mean"),
            duplicate_std=("valuenum", "std")
        )
        .reset_index()
    )

    # 실제 duplicate group만 선택
    duplicate_stats = group_stats.loc[group_stats["n_rows"] > 1].copy()
    
    # ITEMID global SD 연결
    duplicate_stats = duplicate_stats.merge(global_std, on="itemid", how="left")
    duplicate_stats["threshold_5pct"] = 0.05 * duplicate_stats["global_std"]

    # duplicate group의 처리 방법 결정
    duplicate_stats["action"] = "remove"

    exact_mask = duplicate_stats["n_unique_values"] == 1
    duplicate_stats.loc[exact_mask, "action"] = "keep_one"

    conflict_mask = duplicate_stats["n_unique_values"] > 1

    mean_mask = (
        conflict_mask
        & duplicate_stats["global_std"].notna()
        & (duplicate_stats["global_std"] > 0)
        & (duplicate_stats["duplicate_std"] < duplicate_stats["threshold_5pct"])
    )

    duplicate_stats.loc[mean_mask, "action"] = "mean"

    # 각 raw measurement row에 duplicate action 붙이기
    action_info = duplicate_stats[
        group_cols + ["action", "duplicate_mean"]
    ].copy()

    marked = numeric.merge(action_info, on=group_cols, how="left")

    # duplicate가 아닌 정상 measurement
    normal_rows = marked.loc[marked["action"].isna()].copy()

    # 값이 완전히 동일한 duplicate -> 첫번째 measurement만 남기기 
    exact_rows = (
        marked.loc[marked["action"] == "keep_one"]
        .sort_values("_row_order")
        .drop_duplicates(subset=group_cols, keep="first")
        .copy()
    )

    #  5% rule을 통과한 duplicate
    mean_rows = (
        marked.loc[marked["action"] == "mean"]
        .sort_values("_row_order")
        .drop_duplicates(subset=group_cols, keep="first")
        .copy()
    )

    mean_rows["valuenum"] = mean_rows["duplicate_mean"]

    # 최종 numeric data
    cleaned_numeric = pd.concat(
        [normal_rows, exact_rows, mean_rows],
        ignore_index=True
    )

    helper_cols = ["_row_order", "action", "duplicate_mean"]
    cleaned_numeric = cleaned_numeric.drop(
        columns=[c for c in helper_cols if c in cleaned_numeric.columns]
    )
    
    # 15. valuenum이 없던 안건드렸던 row 다시 붙이기
    cleaned = pd.concat(
        [untouched, cleaned_numeric],
        ignore_index=True
    )

    cleaned = cleaned.sort_values(
        [id_col, "charttime", "itemid"]
    ).reset_index(drop=True)


    # 결과 report
    exact_groups = int((duplicate_stats["action"] == "keep_one").sum())
    mean_groups = int((duplicate_stats["action"] == "mean").sum())
    remove_groups = int((duplicate_stats["action"] == "remove").sum())

    report = {
        "table": table_name,

        "numeric_rows_before": int(before_rows),

        "duplicate_groups": int(len(duplicate_stats)),
        "exact_duplicate_groups": exact_groups,
        "mean_resolved_groups": mean_groups,
        "unreliable_removed_groups": remove_groups,

        "numeric_rows_after": int(len(cleaned_numeric)),
        "numeric_rows_removed": int(before_rows - len(cleaned_numeric))
    }

    return cleaned, duplicate_stats, report

# Lactate pool 재생성
# step 2-2에서 Lactate만 따로 모아서 lactate_events를 만들어놨음
# 그런데 step 2-5에서 원본 chartevents / labevents의 duplicate를 처리함
# 기존 lactate_events에는 이 수정 결과가 자동으로 반영되지 않으므로
# 수정된 chartevents / labevents에서 Lactate를 다시 뽑아서 새로 만듦
def _rebuild_lactate_pool(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame):

    # chartevents Lactate
    chart_lactate = chartevents.loc[
        (chartevents["itemid"] == 225668) & chartevents["valuenum"].notna(),
        ["stay_id", "charttime", "valuenum"]
    ].copy()

    chart_lactate["itemid"] = 225668
    chart_lactate["source_table"] = "chartevents"

    # labevents Lactate
    lab_lactate = labevents.loc[
        (labevents["itemid"] == 50813) & labevents["valuenum"].notna(),
        ["hadm_id", "charttime", "valuenum"]
    ].copy()

    stay_windows = stays[["stay_id", "hadm_id", "intime", "outtime"]].copy()

    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"], errors="coerce")
    stay_windows["outtime"] = pd.to_datetime(stay_windows["outtime"], errors="coerce")
    lab_lactate["charttime"] = pd.to_datetime(lab_lactate["charttime"], errors="coerce")

    lab_lactate = lab_lactate.merge(stay_windows, on="hadm_id", how="inner")

    lab_lactate = lab_lactate.loc[
        (lab_lactate["charttime"] >= lab_lactate["intime"])
        & (lab_lactate["charttime"] <= lab_lactate["outtime"])
    ].copy()

    lab_lactate = lab_lactate[["stay_id", "charttime", "valuenum"]]

    lab_lactate["itemid"] = 50813
    lab_lactate["source_table"] = "labevents"

    # 아직 서로 다른 Lactate ITEMID는 merge하지 않는다.
    lactate_events = pd.concat(
        [lab_lactate, chart_lactate],
        ignore_index=True
    )

    lactate_events = lactate_events.sort_values(
        ["stay_id", "charttime", "itemid"]
    ).reset_index(drop=True)

    return lactate_events


# Main
def remove_nonpharma_duplicates(
    stays: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame
):
    # 1. chartevents
    chartevents_clean, chart_duplicate_stats, chart_report = _remove_numeric_duplicates(
        chartevents,
        id_col="stay_id",
        table_name="chartevents"
    )
    
    # 2. labevents
    labevents_clean, lab_duplicate_stats, lab_report = _remove_numeric_duplicates(
        labevents,
        id_col="hadm_id",
        table_name="labevents"
    )

    # 3. Lactate pool도 duplicate 처리 결과를 반영하여 재생성
    lactate_events = _rebuild_lactate_pool(
        stays,
        chartevents_clean,
        labevents_clean
    )

    print("=" * 70)
    print("2-5. Raw Record Duplication Removal")
    print("=" * 70)

    print("[chartevents]")
    print(f"Numeric rows before: {chart_report['numeric_rows_before']}")
    print(f"Duplicate groups: {chart_report['duplicate_groups']}")
    print(f"Exact-value groups → keep one: {chart_report['exact_duplicate_groups']}")
    print(f"Conflict groups → mean: {chart_report['mean_resolved_groups']}")
    print(f"Conflict groups → remove: {chart_report['unreliable_removed_groups']}")
    print(f"Numeric rows removed: {chart_report['numeric_rows_removed']}")
    print(f"Numeric rows after: {chart_report['numeric_rows_after']}")

    print("\n[labevents]")
    print(f"Numeric rows before: {lab_report['numeric_rows_before']}")
    print(f"Duplicate groups: {lab_report['duplicate_groups']}")
    print(f"Exact-value groups → keep one: {lab_report['exact_duplicate_groups']}")
    print(f"Conflict groups → mean: {lab_report['mean_resolved_groups']}")
    print(f"Conflict groups → remove: {lab_report['unreliable_removed_groups']}")
    print(f"Numeric rows removed: {lab_report['numeric_rows_removed']}")
    print(f"Numeric rows after: {lab_report['numeric_rows_after']}")

    print("\n[Lactate pool after 2-5]")
    print(f"Rows: {len(lactate_events)}")

    if len(lactate_events) > 0:
        print(f"Min: {lactate_events['valuenum'].min()}")
        print(f"Median: {lactate_events['valuenum'].median()}")
        print(f"Max: {lactate_events['valuenum'].max()}")

    print("\n[Important]")
    print("All numeric raw ITEMIDs in chartevents/labevents were processed.")
    print("Different ITEMIDs representing the same concept are NOT merged here.")
    print("inputevents/pharmaceutical duplicates are NOT processed here.")

    print("=" * 70)

    return (
        chartevents_clean,
        labevents_clean,
        lactate_events,
        chart_duplicate_stats,
        lab_duplicate_stats,
        chart_report,
        lab_report
    )