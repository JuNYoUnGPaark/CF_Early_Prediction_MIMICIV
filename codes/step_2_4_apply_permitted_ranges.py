import pandas as pd


# ============================================================
# 2-4. Variable-specific Permitted Range
# ============================================================

# Supplementary Table 4에서 permitted range가 명확하게 정의되어 있고,
# 현재 MIMIC-IV ITEMID mapping까지 확정된 chartevents variables
CHARTEVENT_RANGES = {
    220045: ("Heart Rate", 0, 300),

    220050: ("ABP systolic", 10, 300),
    220051: ("ABP diastolic", 10, 175),

    220052: ("ABP mean", 10, 200),
    225312: ("ABP mean", 10, 200),
    224322: ("ABP mean", 10, 200),

    224842: ("Cardiac Output", 1, 20),
    227543: ("Cardiac Output", 1, 20),
    228178: ("Cardiac Output", 1, 20),
    228369: ("Cardiac Output", 1, 20),

    220277: ("SpO2", 10, 100),
    228302: ("RASS", -5, 4),
    224695: ("Ventilator peak pressure", 5, 100),

    225668: ("Lactate", 0, 15),
    227467: ("INR", 0, 8),
    227444: ("C-reactive protein", 0, 600),
}


# Supplementary Table 4에서 permitted range가 명확하게 정의되어 있고,
# 현재 MIMIC-IV ITEMID mapping까지 확정된 labevents variables
LABEVENT_RANGES = {
    50813: ("Lactate", 0, 15),
    51237: ("INR", 0, 8),
    50889: ("C-reactive protein", 0, 600),
}


# ============================================================
# ITEMID별 permitted range 적용
# ============================================================

def _filter_table(df: pd.DataFrame, rules: dict, table_name: str):

    filtered = df.copy()
    remove_mask = pd.Series(False, index=filtered.index)
    report_rows = []

    for itemid, (variable, lower, upper) in rules.items():

        # 현재 ITEMID이면서 실제 numerical value가 있는 measurement
        target_mask = (filtered["itemid"] == itemid) & filtered["valuenum"].notna()

        # permitted range 밖에 있는 measurement
        out_of_range_mask = target_mask & ~filtered["valuenum"].between(lower, upper, inclusive="both")

        before = int(target_mask.sum())
        removed = int(out_of_range_mask.sum())
        after = before - removed

        remove_mask |= out_of_range_mask

        report_rows.append({
            "table": table_name,
            "itemid": itemid,
            "variable": variable,
            "permitted_range": f"[{lower}, {upper}]",
            "before": before,
            "removed": removed,
            "after": after
        })

    # 범위를 벗어난 measurement row만 삭제
    filtered = filtered.loc[~remove_mask].copy().reset_index(drop=True)

    return filtered, pd.DataFrame(report_rows)


# ============================================================
# Main
# ============================================================

def apply_permitted_ranges(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame, lactate_events: pd.DataFrame):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------

    if "stay_id" not in stays.columns:
        raise ValueError("stays에 stay_id column이 없습니다.")

    required_chart = {"stay_id", "itemid", "valuenum"}
    missing = required_chart - set(chartevents.columns)

    if missing:
        raise ValueError(f"chartevents에 필요한 column이 없습니다: {missing}")

    required_lab = {"hadm_id", "itemid", "charttime", "valuenum"}
    missing = required_lab - set(labevents.columns)

    if missing:
        raise ValueError(f"labevents에 필요한 column이 없습니다: {missing}")


    # --------------------------------------------------------
    # 2. 현재 preprocessing cohort 확인
    # --------------------------------------------------------
    # 현재 1-3, 1-4, 1-5를 통과한 2878 stays만 사용한다.
    #
    # 이전 코드에서는 raw chartevents/labevents 전체 3000 subset에
    # range를 적용했기 때문에 lactate_events(2878 stays)와
    # measurement count 기준이 달랐다.
    #
    # 이번에는 모든 table을 현재 2878 stays 기준으로 맞춘다.
    # --------------------------------------------------------

    current_stay_ids = set(stays["stay_id"])


    # --------------------------------------------------------
    # 3. chartevents → 현재 2878 stays만 남기기
    # --------------------------------------------------------
    # chartevents에는 stay_id가 직접 있으므로 간단히 filtering 가능
    # --------------------------------------------------------

    chartevents = chartevents.loc[chartevents["stay_id"].isin(current_stay_ids)].copy().reset_index(drop=True)


    # --------------------------------------------------------
    # 4. labevents → 현재 2878 ICU stays에 해당하는 기록만 남기기
    # --------------------------------------------------------
    # labevents에는 stay_id가 없고 hadm_id만 있다.
    #
    # 따라서:
    #
    #   같은 hadm_id
    #   AND
    #   ICU intime <= lab charttime <= ICU outtime
    #
    # 인 measurement만 현재 ICU stay의 lab으로 사용한다.
    #
    # _row_id를 사용하는 이유:
    # merge 과정에서 원래 labevents row가 중복되는 것을 방지하고
    # 최종적으로 원래 labevents 구조를 유지하기 위해서이다.
    # --------------------------------------------------------

    stay_windows = stays[["hadm_id", "intime", "outtime"]].copy()
    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"], errors="coerce")
    stay_windows["outtime"] = pd.to_datetime(stay_windows["outtime"], errors="coerce")

    labevents = labevents.copy()
    labevents["charttime"] = pd.to_datetime(labevents["charttime"], errors="coerce")
    labevents["_row_id"] = range(len(labevents))

    lab_match = labevents.merge(stay_windows, on="hadm_id", how="inner")

    lab_match = lab_match.loc[
        (lab_match["charttime"] >= lab_match["intime"])
        & (lab_match["charttime"] <= lab_match["outtime"])
    ].copy()

    valid_lab_row_ids = set(lab_match["_row_id"].unique())

    labevents = labevents.loc[labevents["_row_id"].isin(valid_lab_row_ids)].copy()
    labevents = labevents.drop(columns="_row_id").reset_index(drop=True)


    # --------------------------------------------------------
    # 5. chartevents permitted range 적용
    # --------------------------------------------------------

    filtered_chartevents, chart_report = _filter_table(
        chartevents,
        CHARTEVENT_RANGES,
        "chartevents"
    )


    # --------------------------------------------------------
    # 6. labevents permitted range 적용
    # --------------------------------------------------------

    filtered_labevents, lab_report = _filter_table(
        labevents,
        LABEVENT_RANGES,
        "labevents"
    )


    # --------------------------------------------------------
    # 7. 이전 2-2에서 만든 Lactate pool에도 [0,15] 적용
    # --------------------------------------------------------
    # lactate_events에는:
    #
    #   labevents 50813
    #   chartevents 225668
    #
    # 두 Lactate source가 이미 합쳐져 있다.
    #
    # 여기에도 같은 permitted range를 적용해서
    # 이후 State Annotation 등에 바로 사용할 수 있도록 만든다.
    # --------------------------------------------------------

    lactate_before = len(lactate_events)

    lactate_bad = (
        lactate_events["valuenum"].notna()
        & ~lactate_events["valuenum"].between(0, 15, inclusive="both")
    )

    filtered_lactate_events = lactate_events.loc[~lactate_bad].copy().reset_index(drop=True)

    lactate_removed = int(lactate_bad.sum())
    lactate_after = len(filtered_lactate_events)


    # --------------------------------------------------------
    # 8. ITEMID별 report
    # --------------------------------------------------------

    range_report = pd.concat([chart_report, lab_report], ignore_index=True)


    # --------------------------------------------------------
    # 9. Variable 단위 report
    # --------------------------------------------------------
    # 예:
    #
    # Lactate:
    #   225668 + 50813
    #
    # ABP mean:
    #   220052 + 225312 + 224322
    #
    # 를 하나의 variable로 합산해서 보여준다.
    # --------------------------------------------------------

    variable_report = (
        range_report.groupby("variable", as_index=False)[["before", "removed", "after"]]
        .sum()
        .sort_values("variable")
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # 10. 전체 report
    # --------------------------------------------------------

    report = {
        "current_stays": int(len(stays)),
        "chartevents_removed": int(chart_report["removed"].sum()),
        "labevents_removed": int(lab_report["removed"].sum()),
        "total_removed": int(range_report["removed"].sum()),
        "lactate_before": int(lactate_before),
        "lactate_removed": int(lactate_removed),
        "lactate_after": int(lactate_after)
    }


    # --------------------------------------------------------
    # 11. 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("2-4. Variable-specific Permitted Range")
    print("=" * 70)

    print(f"Current stays: {report['current_stays']}")
    print(f"Removed from chartevents: {report['chartevents_removed']}")
    print(f"Removed from labevents: {report['labevents_removed']}")
    print(f"Total out-of-range rows removed: {report['total_removed']}")

    print("\n[Variable-level removal]")
    print(variable_report.to_string(index=False))

    print("\n[Lactate pool]")
    print(f"Before: {report['lactate_before']}")
    print(f"Removed: {report['lactate_removed']}")
    print(f"After: {report['lactate_after']}")

    if len(filtered_lactate_events) > 0:
        print(f"Min after filtering: {filtered_lactate_events['valuenum'].min()}")
        print(f"Median after filtering: {filtered_lactate_events['valuenum'].median()}")
        print(f"Max after filtering: {filtered_lactate_events['valuenum'].max()}")

    print("\n[Important]")
    print("Only measurements outside permitted ranges are removed.")
    print("ICU stays are NOT excluded in this step.")
    print("static_Height permitted range = all → Height is not filtered here.")
    print("Unmapped variables are not filtered until mapping is finalized.")

    print("=" * 70)

    return filtered_chartevents, filtered_labevents, filtered_lactate_events, range_report, variable_report, report