import pandas as pd


# ============================================================
# 4-1A. Merge Identical Physiology / Lab Concepts
# ============================================================
#
# 서로 다른 MIMIC-IV ITEMID라도 동일한 clinical concept이면
# 하나의 canonical variable 이름으로 통합한다.
#
# 이 단계에서는 같은 시각의 여러 값을 아직 median 처리하지 않는다.
# → 그 처리는 4-2에서 수행한다.
#
# source_table / source_itemid는 그대로 남겨서
# 원래 어디에서 온 값인지 추적할 수 있도록 한다.
# ============================================================


CHARTEVENT_MAP = {
    220045: "Heart Rate",

    220050: "ABP systolic",
    220051: "ABP diastolic",

    220052: "ABP mean",
    225312: "ABP mean",
    224322: "ABP mean",       # IABP source → 나중 State Annotation에서 제외

    224842: "Cardiac Output",
    227543: "Cardiac Output",
    228178: "Cardiac Output",
    228369: "Cardiac Output",

    220277: "SpO2",
    228302: "RASS",
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


def merge_nonpharma_concepts(
    stays,
    chartevents,
    labevents
):

    # ========================================================
    # 1. CHARTEVENTS
    # ========================================================

    current_stays = set(stays["stay_id"])

    chart = chartevents[
        chartevents["stay_id"].isin(current_stays)
        & chartevents["itemid"].isin(CHARTEVENT_MAP)
        & chartevents["valuenum"].notna()
    ].copy()

    chart["charttime"] = pd.to_datetime(
        chart["charttime"],
        errors="coerce"
    )

    chart["variable"] = chart["itemid"].map(
        CHARTEVENT_MAP
    )

    chart["source_table"] = "chartevents"
    chart["source_itemid"] = chart["itemid"]

    chart = chart[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
            "source_table",
            "source_itemid",
        ]
    ]


    # ========================================================
    # 2. LABEVENTS
    # ========================================================
    #
    # labevents에는 stay_id가 없으므로:
    #
    # hadm_id
    # + charttime이 ICU intime~outtime 안에 있는지
    #
    # 를 이용해 stay_id를 연결한다.
    # ========================================================

    lab = labevents[
        labevents["itemid"].isin(LABEVENT_MAP)
        & labevents["valuenum"].notna()
    ].copy()

    lab["charttime"] = pd.to_datetime(
        lab["charttime"],
        errors="coerce"
    )

    lab["_lab_row_id"] = range(len(lab))

    stay_times = stays[
        ["stay_id", "hadm_id", "intime", "outtime"]
    ].copy()

    stay_times["intime"] = pd.to_datetime(
        stay_times["intime"]
    )

    stay_times["outtime"] = pd.to_datetime(
        stay_times["outtime"]
    )

    lab = lab.merge(
        stay_times,
        on="hadm_id",
        how="inner"
    )

    lab = lab[
        (lab["charttime"] >= lab["intime"])
        & (lab["charttime"] <= lab["outtime"])
    ].copy()


    # 같은 lab row가 둘 이상의 ICU stay에 들어가는지 확인
    ambiguous_lab_rows = (
        lab.groupby("_lab_row_id")["stay_id"]
        .nunique()
        .gt(1)
        .sum()
    )


    lab["variable"] = lab["itemid"].map(
        LABEVENT_MAP
    )

    lab["source_table"] = "labevents"
    lab["source_itemid"] = lab["itemid"]

    lab = lab[
        [
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
            "source_table",
            "source_itemid",
        ]
    ]


    # ========================================================
    # 3. 두 source 합치기
    # ========================================================

    events = pd.concat(
        [chart, lab],
        ignore_index=True
    )

    events = events.sort_values(
        ["stay_id", "charttime", "variable"]
    ).reset_index(drop=True)


    # ========================================================
    # 4. 확인용 report
    # ========================================================

    report = (
        events
        .groupby("variable")
        .agg(
            rows=("valuenum", "size"),
            stays=("stay_id", "nunique"),
            source_itemids=("source_itemid", "nunique"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )


    print("=" * 70)
    print("4-1A. Physiology / Lab Concept Merging")
    print("=" * 70)

    print(f"Canonical variables: {events['variable'].nunique()}")
    print(f"Total mapped rows: {len(events)}")
    print(f"Ambiguous lab→ICU mappings: {ambiguous_lab_rows}")

    print("\n[Variable summary]")
    print(report.to_string(index=False))


    print("\n[ITEMID → canonical variable]")

    mapping_report = (
        events[
            [
                "source_table",
                "source_itemid",
                "variable"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["variable", "source_table", "source_itemid"]
        )
    )

    print(mapping_report.to_string(index=False))


    print("\n[Important]")
    print("No simultaneous measurements were aggregated yet.")
    print("source_itemid is retained for later provenance-specific rules.")
    print("IABP-derived ABP mean (ITEMID 224322) is retained here,")
    print("but will be excluded from MAP used for CF state annotation.")

    print("=" * 70)


    return events, report, mapping_report