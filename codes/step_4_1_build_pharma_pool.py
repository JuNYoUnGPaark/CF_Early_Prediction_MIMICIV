import pandas as pd


# ============================================================
# 4-1C. Build Final Pharmaceutical Pool
# ============================================================
#
# Table4와 정확히 연결된 MIMIC-IV ITEMID만 사용한다.
#
# continuous:
#   3-2에서 만든 continuous_rate 사용
#
# non-continuous:
#   3-3에서 만든 acting-period effective_rate 사용
#
# 같은 Table4 group에 속하는 약물은 동일한
# canonical pharmaceutical variable 이름을 부여한다.
#
# 아직 rate를 sum/merge하지 않는다.
# ============================================================


def build_pharma_pool(
    continuous_events,
    noncontinuous_effective_events,
    pharma_target_candidates
):

    # --------------------------------------------------------
    # 1. ITEMID당 Table4 mapping 하나만 선택
    # --------------------------------------------------------
    #
    # Propofol / Norepinephrine처럼
    # exact_constituent + exact_group 둘 다 잡힌 경우가 있으므로
    # constituent를 우선하고 ITEMID당 한 줄만 남긴다.
    # --------------------------------------------------------

    mapping = pharma_target_candidates[
        pharma_target_candidates["table4_group"].notna()
    ].copy()

    priority = {
        "exact_constituent": 0,
        "exact_group": 1,
    }

    mapping["priority"] = (
        mapping["match_type"]
        .map(priority)
        .fillna(9)
    )

    mapping = (
        mapping
        .sort_values(["itemid", "priority"])
        .drop_duplicates("itemid")
        .copy()
    )

    # 최종 pharmaceutical variable
    mapping["pharma_variable"] = mapping["table4_group"]


    # --------------------------------------------------------
    # 2. Continuous
    # --------------------------------------------------------

    cont = continuous_events[
        continuous_events["itemid"].isin(mapping["itemid"])
    ].copy()

    cont = cont.merge(
        mapping[["itemid", "pharma_variable"]],
        on="itemid",
        how="left"
    )

    cont["rate_value"] = cont["continuous_rate"]
    cont["rate_uom_final"] = cont["continuous_rate_uom"]

    cont["effective_starttime"] = pd.to_datetime(
        cont["starttime"]
    )

    cont["effective_endtime"] = pd.to_datetime(
        cont["endtime"]
    )

    cont["administration_form"] = "continuous"


    # --------------------------------------------------------
    # 3. Non-continuous
    # --------------------------------------------------------

    noncont = noncontinuous_effective_events[
        noncontinuous_effective_events["itemid"].isin(mapping["itemid"])
    ].copy()

    noncont = noncont.merge(
        mapping[["itemid", "pharma_variable"]],
        on="itemid",
        how="left"
    )

    noncont["rate_value"] = noncont["effective_rate"]
    noncont["rate_uom_final"] = noncont["effective_rate_uom"]

    noncont["administration_form"] = "non-continuous"


    # --------------------------------------------------------
    # 4. 같은 형식으로 합치기
    # --------------------------------------------------------

    cols = [
        "stay_id",
        "itemid",
        "label",
        "pharma_variable",
        "administration_form",
        "effective_starttime",
        "effective_endtime",
        "rate_value",
        "rate_uom_final",
    ]

    pharma_events = pd.concat(
        [
            cont[cols],
            noncont[cols],
        ],
        ignore_index=True
    )

    pharma_events = pharma_events.sort_values(
        [
            "stay_id",
            "effective_starttime",
            "pharma_variable",
        ]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    summary = (
        pharma_events
        .groupby("pharma_variable")
        .agg(
            rows=("itemid", "size"),
            itemids=("itemid", "nunique"),
            stays=("stay_id", "nunique"),
            units=("rate_uom_final", "nunique"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )


    print("=" * 70)
    print("4-1C. Final Pharmaceutical Pool")
    print("=" * 70)

    print(f"Mapped ITEMIDs: {mapping['itemid'].nunique()}")
    print(f"Canonical pharma variables: {pharma_events['pharma_variable'].nunique()}")

    print(
        "Continuous rows:",
        len(cont)
    )

    print(
        "Non-continuous rows:",
        len(noncont)
    )

    print(
        "Total pharma rows:",
        len(pharma_events)
    )


    print("\n[Pharmaceutical variables]")
    print(summary.to_string(index=False))


    print("\n[ITEMID → pharmaceutical variable]")

    print(
        mapping[
            [
                "itemid",
                "label",
                "table4_group",
                "table4_drug",
                "match_type",
            ]
        ]
        .sort_values(["table4_group", "itemid"])
        .to_string(index=False)
    )


    print("\n[Important]")
    print("No pharmaceutical rates were summed yet.")
    print("Only canonical pharmaceutical variable names were assigned.")
    print("Simultaneous-event aggregation is performed in 4-2.")

    print("=" * 70)


    return pharma_events, mapping, summary