import pandas as pd

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
    220088: ("Cardiac Output", 1, 20),

    220277: ("SpO2", 10, 100),
    228096: ("RASS", -5, 4),
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

# ITEMID별 permitted range 적용
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
        
        # 지금까지 발견한 모든 out-of-range row를 하나의 remove_mask에 계속 누적
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


def apply_permitted_ranges(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame, lactate_events: pd.DataFrame):

    current_stay_ids = set(stays["stay_id"])

    chartevents = chartevents.loc[chartevents["stay_id"].isin(current_stay_ids)].copy().reset_index(drop=True)


    # labevents → 현재 2878 ICU stays에 해당하는 기록만 남기기
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


    # chartevents permitted range 적용
    filtered_chartevents, chart_report = _filter_table(
        chartevents,
        CHARTEVENT_RANGES,
        "chartevents"
    )

    # labevents permitted range 적용
    filtered_labevents, lab_report = _filter_table(
        labevents,
        LABEVENT_RANGES,
        "labevents"
    )


    #  Lactate pool에도 [0,15] 적용
    lactate_before = len(lactate_events)

    lactate_bad = (
        lactate_events["valuenum"].notna()
        & ~lactate_events["valuenum"].between(0, 15, inclusive="both")
    )

    filtered_lactate_events = lactate_events.loc[~lactate_bad].copy().reset_index(drop=True)
    lactate_removed = int(lactate_bad.sum())
    lactate_after = len(filtered_lactate_events)

    #  ITEMID별 report
    range_report = pd.concat([chart_report, lab_report], ignore_index=True)
    # Variable 단위 report
    variable_report = (
        range_report.groupby("variable", as_index=False)[["before", "removed", "after"]]
        .sum()
        .sort_values("variable")
        .reset_index(drop=True)
    )
    report = {
        "current_stays": int(len(stays)),
        "chartevents_removed": int(chart_report["removed"].sum()),
        "labevents_removed": int(lab_report["removed"].sum()),
        "total_removed": int(range_report["removed"].sum()),
        "lactate_before": int(lactate_before),
        "lactate_removed": int(lactate_removed),
        "lactate_after": int(lactate_after)
    }

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