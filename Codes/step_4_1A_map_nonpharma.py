import pandas as pd

# 현재까지 mapping이 명확하게 확인된 chartevents ITEMID
CHARTEVENT_MAP = {
    220045: "Heart Rate",

    220050: "ABP systolic",
    220051: "ABP diastolic",

    220052: "ABP mean",
    225312: "ABP mean",
    224322: "ABP mean",   # IABP-derived source. provenance 보존 후 CF MAP에서 제외 가능

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


# 현재까지 mapping이 명확하게 확인된 labevents ITEMID
LABEVENT_MAP = {
    50813: "Lactate",

    51237: "INR",

    50931: "Blood Glucose",
    50809: "Blood Glucose",

    50889: "C-reactive protein",
}


def merge_nonpharma_concepts(
    stays: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
):
    # CHARTEVENTS
    current_stay_ids = set(stays["stay_id"].dropna().unique())

    chart = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin(CHARTEVENT_MAP)
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "charttime", "valuenum"]
    ].copy()

    chart["charttime"] = pd.to_datetime(
        chart["charttime"],
        errors="coerce"
    )

    chart = chart.dropna(subset=["charttime"]).copy()

    chart["variable"] = chart["itemid"].map(CHARTEVENT_MAP)

    chart_glucose_mask = chart["itemid"].isin(
        [220621, 225664, 226537]
    )
    chart.loc[chart_glucose_mask, "valuenum"] = (
        chart.loc[chart_glucose_mask, "valuenum"] / 18.0
    )

    chart["source_table"] = "chartevents"
    chart["source_itemid"] = chart["itemid"]

    # 224322 = IABP mean source.
    # 일반 ABP mean concept에는 보존하지만,
    # 이후 CF state annotation용 MAP에서는 제외할 수 있도록 flag를 남긴다.
    chart["is_cf_map_source"] = True
    chart.loc[
        chart["source_itemid"] == 224322,
        "is_cf_map_source"
    ] = False

    chart = chart[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
            "source_table",
            "source_itemid",
            "is_cf_map_source",
        ]
    ]

    # LABEVENTS
    lab = labevents.loc[
        labevents["itemid"].isin(LABEVENT_MAP)
        & labevents["valuenum"].notna(),
        ["hadm_id", "itemid", "charttime", "valuenum"]
    ].copy()

    lab["charttime"] = pd.to_datetime(
        lab["charttime"],
        errors="coerce"
    )

    lab = lab.dropna(subset=["charttime"]).copy()
    lab["_lab_row_id"] = range(len(lab))

    stay_windows = stays[
        ["stay_id", "hadm_id", "intime", "outtime"]
    ].copy()

    stay_windows["intime"] = pd.to_datetime(
        stay_windows["intime"],
        errors="coerce"
    )

    stay_windows["outtime"] = pd.to_datetime(
        stay_windows["outtime"],
        errors="coerce"
    )

    lab_match = lab.merge(
        stay_windows,
        on="hadm_id",
        how="inner"
    )

    lab_match = lab_match.loc[
        (lab_match["charttime"] >= lab_match["intime"])
        & (lab_match["charttime"] <= lab_match["outtime"])
    ].copy()


    # 하나의 lab row가 둘 이상의 ICU stay에 들어가는지 확인
    lab_to_stay_count = (
        lab_match
        .groupby("_lab_row_id")["stay_id"]
        .nunique()
    )

    ambiguous_lab_row_ids = set(
        lab_to_stay_count[
            lab_to_stay_count > 1
        ].index
    )

    # 애매한 lab row는 자동 배정하지 않는다.
    if ambiguous_lab_row_ids:
        lab_match = lab_match.loc[
            ~lab_match["_lab_row_id"].isin(ambiguous_lab_row_ids)
        ].copy()

    # lab canonical variable 부여
    lab_match["variable"] = (
        lab_match["itemid"].map(LABEVENT_MAP)
    )

    # MIMIC author mapping special handling:
    # 50931, 50809 Glucose mg/dL -> mmol/L
    lab_glucose_mask = lab_match["itemid"].isin(
        [50931, 50809]
    )
    lab_match.loc[lab_glucose_mask, "valuenum"] = (
        lab_match.loc[lab_glucose_mask, "valuenum"] / 18.0
    )

    lab_match["source_table"] = "labevents"
    lab_match["source_itemid"] = lab_match["itemid"]
    lab_match["is_cf_map_source"] = True

    lab_final = lab_match[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
            "source_table",
            "source_itemid",
            "is_cf_map_source",
        ]
    ].copy()

    # CHARTEVENTS + LABEVENTS
    events = pd.concat(
        [chart, lab_final],
        ignore_index=True
    )

    events = events.sort_values(
        [
            "stay_id",
            "charttime",
            "variable",
            "source_table",
            "source_itemid",
        ]
    ).reset_index(drop=True)

    variable_report = (
        events
        .groupby("variable")
        .agg(
            rows=("valuenum", "size"),
            stays=("stay_id", "nunique"),
            source_itemids=("source_itemid", "nunique"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    
    mapping_report = (
        events[
            [
                "source_table",
                "source_itemid",
                "variable",
                "is_cf_map_source",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "variable",
                "source_table",
                "source_itemid",
            ]
        )
        .reset_index(drop=True)
    )

    report = {
        "current_stays": int(len(stays)),
        "canonical_variables": int(events["variable"].nunique()),
        "mapped_rows": int(len(events)),
        "chartevent_rows": int(len(chart)),
        "labevent_rows": int(len(lab_final)),
        "ambiguous_lab_rows_removed": int(len(ambiguous_lab_row_ids)),
    }

    print("=" * 70)
    print("4-1A. Non-pharmaceutical Concept Mapping")
    print("=" * 70)

    print(f"Current stays: {report['current_stays']}")
    print(f"Canonical variables: {report['canonical_variables']}")
    print(f"Mapped rows: {report['mapped_rows']}")
    print(f"  chartevents: {report['chartevent_rows']}")
    print(f"  labevents: {report['labevent_rows']}")
    print(
        "Ambiguous lab rows removed: "
        f"{report['ambiguous_lab_rows_removed']}"
    )


    print("\n[Variable summary]")
    print(
        variable_report.to_string(index=False)
    )


    print("\n[ITEMID → canonical variable]")
    print(
        mapping_report.to_string(index=False)
    )


    print("\n[Important]")
    print("Only explicitly confirmed ITEMID mappings are used.")
    print("No fuzzy/synonym matching is performed.")
    print("No simultaneous measurements are aggregated in this step.")
    print("source_itemid is preserved for provenance-specific rules.")
    print("IABP mean ITEMID 224322 is flagged is_cf_map_source=False.")
    print("Blood Glucose MIMIC values were converted mg/dL -> mmol/L (/18).")

    glucose = events.loc[events["variable"] == "Blood Glucose", "valuenum"]
    if len(glucose):
        print(
            "Blood Glucose after conversion "
            f"(min/median/max): {glucose.min():.3f} / "
            f"{glucose.median():.3f} / {glucose.max():.3f}"
        )

    print("=" * 70)


    return (
        events,
        variable_report,
        mapping_report,
        report,
    )