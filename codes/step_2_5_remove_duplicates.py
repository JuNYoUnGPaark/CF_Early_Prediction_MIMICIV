import numpy as np
import pandas as pd


# ============================================================
# 2-5. Raw Record Duplication Removal
# ============================================================
#
# 대상:
#   - chartevents: valuenum이 존재하는 모든 raw ITEMID
#   - labevents  : valuenum이 존재하는 모든 raw ITEMID
#
# 제외:
#   - inputevents (pharmaceutical variable)
#     → 이후 Pharmaceutical Processing에서 별도로 처리
#
#
# 원 논문 duplicate rule
# ------------------------------------------------------------
# 같은 patient + variable + timestamp에 여러 값이 존재하면:
#
# 1) 값이 모두 동일
#       → 하나만 유지
#
# 2) 값이 서로 다름
#       → duplicate group SD 계산
#       → 해당 raw variable(ITEMID)의 global SD와 비교
#
#       duplicate SD < 0.05 × global SD
#           → duplicate들의 평균값 하나를 유지
#
#       duplicate SD >= 0.05 × global SD
#           → unreliable하므로 해당 group 전부 삭제
#
#
# 중요:
# 여기서 variable은 아직 최종 clinical concept이 아니라
# MIMIC-IV의 raw ITEMID이다.
#
# 예:
#   ITEMID 50813과 ITEMID 225668은 둘 다 Lactate이지만
#   여기서는 서로 다른 raw variable로 처리한다.
#
# 두 ITEMID를 하나의 Lactate로 합치는 것은
# 이후 Variable Merging 단계에서 수행한다.
# ============================================================


def _remove_numeric_duplicates(df: pd.DataFrame, id_col: str, table_name: str):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------

    required = {id_col, "itemid", "charttime", "valuenum"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{table_name}에 필요한 column이 없습니다: {missing}")


    # --------------------------------------------------------
    # 2. 복사 + 시간 형식 정리
    # --------------------------------------------------------

    work = df.copy()
    work["charttime"] = pd.to_datetime(work["charttime"], errors="coerce")


    # --------------------------------------------------------
    # 3. numeric measurement와 나머지 row 분리
    # --------------------------------------------------------
    #
    # duplicate SD를 계산해야 하므로 valuenum이 실제로 존재하는
    # measurement만 이번 단계의 대상이다.
    #
    # valuenum이 없는 text/categorical record는 삭제하지 않고
    # untouched에 그대로 보존한다.
    # --------------------------------------------------------

    target_mask = work["valuenum"].notna() & work["charttime"].notna()

    numeric = work.loc[target_mask].copy()
    untouched = work.loc[~target_mask].copy()

    numeric["_row_order"] = np.arange(len(numeric))

    before_rows = len(numeric)


    # --------------------------------------------------------
    # 4. raw ITEMID별 global SD 계산
    # --------------------------------------------------------
    #
    # 원 논문의:
    # "global standard deviation of the variable across all patients"
    #
    # 를 MIMIC-IV에서는 각 raw ITEMID의 전체 measurement SD로
    # 구현한다.
    #
    # duplicate SD와 global SD 모두 동일하게 pandas std()
    # 즉 sample standard deviation(ddof=1)을 사용한다.
    # --------------------------------------------------------

    global_std = (
        numeric.groupby("itemid")["valuenum"]
        .std()
        .rename("global_std")
        .reset_index()
    )


    # --------------------------------------------------------
    # 5. same patient/stay + ITEMID + timestamp 기준 group 생성
    # --------------------------------------------------------
    #
    # chartevents:
    #   stay_id + itemid + charttime
    #
    # labevents:
    #   hadm_id + itemid + charttime
    #
    # labevents는 앞의 2-4에서 이미 현재 ICU time window 안의
    # measurement만 남겨놓은 상태이다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 6. 실제 duplicate group만 선택
    # --------------------------------------------------------

    duplicate_stats = group_stats.loc[group_stats["n_rows"] > 1].copy()


    # --------------------------------------------------------
    # 7. ITEMID global SD 연결
    # --------------------------------------------------------

    duplicate_stats = duplicate_stats.merge(global_std, on="itemid", how="left")
    duplicate_stats["threshold_5pct"] = 0.05 * duplicate_stats["global_std"]


    # --------------------------------------------------------
    # 8. duplicate group의 처리 방법 결정
    # --------------------------------------------------------
    #
    # exact:
    #   값이 전부 동일 → 하나만 유지
    #
    # mean:
    #   서로 다른 값이지만 duplicate SD가 global SD의 5% 미만
    #   → 평균값 하나 유지
    #
    # remove:
    #   값 차이가 큼
    #   → unreliable → 모두 삭제
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 9. 각 raw measurement row에 duplicate action 붙이기
    # --------------------------------------------------------

    action_info = duplicate_stats[
        group_cols + ["action", "duplicate_mean"]
    ].copy()

    marked = numeric.merge(action_info, on=group_cols, how="left")


    # --------------------------------------------------------
    # 10. duplicate가 아닌 정상 measurement
    # --------------------------------------------------------

    normal_rows = marked.loc[marked["action"].isna()].copy()


    # --------------------------------------------------------
    # 11. 값이 완전히 동일한 duplicate
    # --------------------------------------------------------
    #
    # 첫 번째 measurement 하나만 유지
    # --------------------------------------------------------

    exact_rows = (
        marked.loc[marked["action"] == "keep_one"]
        .sort_values("_row_order")
        .drop_duplicates(subset=group_cols, keep="first")
        .copy()
    )


    # --------------------------------------------------------
    # 12. 5% rule을 통과한 conflicting duplicate
    # --------------------------------------------------------
    #
    # group에서 첫 번째 row의 metadata를 대표로 사용하고
    # valuenum만 duplicate들의 평균값으로 교체한다.
    # --------------------------------------------------------

    mean_rows = (
        marked.loc[marked["action"] == "mean"]
        .sort_values("_row_order")
        .drop_duplicates(subset=group_cols, keep="first")
        .copy()
    )

    mean_rows["valuenum"] = mean_rows["duplicate_mean"]


    # --------------------------------------------------------
    # 13. action == remove인 group은 아무것도 추가하지 않는다.
    # --------------------------------------------------------
    #
    # 즉 해당 patient-variable-time의 measurement들이
    # 모두 제거된다.
    # --------------------------------------------------------


    # --------------------------------------------------------
    # 14. 최종 numeric data
    # --------------------------------------------------------

    cleaned_numeric = pd.concat(
        [normal_rows, exact_rows, mean_rows],
        ignore_index=True
    )

    helper_cols = ["_row_order", "action", "duplicate_mean"]
    cleaned_numeric = cleaned_numeric.drop(
        columns=[c for c in helper_cols if c in cleaned_numeric.columns]
    )


    # --------------------------------------------------------
    # 15. valuenum이 없던 untouched row 다시 붙이기
    # --------------------------------------------------------

    cleaned = pd.concat(
        [untouched, cleaned_numeric],
        ignore_index=True
    )

    cleaned = cleaned.sort_values(
        [id_col, "charttime", "itemid"]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 16. 결과 report
    # --------------------------------------------------------

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


# ============================================================
# Lactate pool 재생성
# ============================================================

def _rebuild_lactate_pool(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame):

    # --------------------------------------------------------
    # chartevents Lactate
    # --------------------------------------------------------

    chart_lactate = chartevents.loc[
        (chartevents["itemid"] == 225668) & chartevents["valuenum"].notna(),
        ["stay_id", "charttime", "valuenum"]
    ].copy()

    chart_lactate["itemid"] = 225668
    chart_lactate["source_table"] = "chartevents"


    # --------------------------------------------------------
    # labevents Lactate
    # --------------------------------------------------------
    #
    # labevents에는 stay_id가 없으므로
    # hadm_id + ICU intime/outtime으로 stay_id를 다시 붙인다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 아직 서로 다른 Lactate ITEMID는 merge하지 않는다.
    # --------------------------------------------------------

    lactate_events = pd.concat(
        [lab_lactate, chart_lactate],
        ignore_index=True
    )

    lactate_events = lactate_events.sort_values(
        ["stay_id", "charttime", "itemid"]
    ).reset_index(drop=True)

    return lactate_events


# ============================================================
# Main
# ============================================================

def remove_nonpharma_duplicates(
    stays: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame
):

    # --------------------------------------------------------
    # 1. chartevents
    # --------------------------------------------------------

    chartevents_clean, chart_duplicate_stats, chart_report = _remove_numeric_duplicates(
        chartevents,
        id_col="stay_id",
        table_name="chartevents"
    )


    # --------------------------------------------------------
    # 2. labevents
    # --------------------------------------------------------

    labevents_clean, lab_duplicate_stats, lab_report = _remove_numeric_duplicates(
        labevents,
        id_col="hadm_id",
        table_name="labevents"
    )


    # --------------------------------------------------------
    # 3. Lactate pool도 duplicate 처리 결과를 반영하여 재생성
    # --------------------------------------------------------

    lactate_events = _rebuild_lactate_pool(
        stays,
        chartevents_clean,
        labevents_clean
    )


    # --------------------------------------------------------
    # 4. 결과 출력
    # --------------------------------------------------------

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