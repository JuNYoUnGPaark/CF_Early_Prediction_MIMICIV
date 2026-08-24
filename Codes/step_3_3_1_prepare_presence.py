import pandas as pd


# ============================================================
# 3-3-1. Prepare Binary Presence Candidates
# ============================================================
#
# 원 논문:
# quantitative rate로 표현하기 어려운 경우
# drug / drug class가 존재하는지를 binary flag로 표현
#
# MIMIC-IV:
# acting period를 적용한 17개 ITEMID는 제외하고,
# 나머지 non-continuous events를 presence 후보로 만든다.
#
# 주의:
# 여기서 128개 ITEMID를 모두 최종 약물 변수로 확정하지 않는다.
# 실제 drug / drug class 선택 및 merging은 이후 단계에서 수행한다.
# ============================================================


def prepare_noncontinuous_presence(
    inputevents_classified,
    acting_period_candidates
):

    # --------------------------------------------------------
    # 1. Acting period가 확인된 17개 ITEMID
    # --------------------------------------------------------

    quantitative_itemids = set(
        acting_period_candidates.loc[
            acting_period_candidates["mapping_status"] == "exact_name_match",
            "itemid"
        ]
    )


    # --------------------------------------------------------
    # 2. Non-continuous events
    # --------------------------------------------------------

    events = inputevents_classified[
        inputevents_classified["administration_type"]
        == "non-continuous"
    ].copy()


    # --------------------------------------------------------
    # 3. 이미 effective rate로 바꾼 17개는 제외
    # --------------------------------------------------------

    presence = events[
        ~events["itemid"].isin(quantitative_itemids)
    ].copy()


    # --------------------------------------------------------
    # 4. Binary presence
    # --------------------------------------------------------
    #
    # 아직 5-minute grid를 만들지 않았으므로
    # event 단위로 presence=1만 기록한다.
    #
    # starttime / endtime은 그대로 보존한다.
    # 이후 time-grid 생성 시 해당 interval과 겹치는 시점에
    # presence=1을 적용할 수 있다.
    # --------------------------------------------------------

    presence["presence"] = 1

    presence["presence_starttime"] = pd.to_datetime(
        presence["starttime"],
        errors="coerce"
    )

    presence["presence_endtime"] = pd.to_datetime(
        presence["endtime"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    summary = (
        presence
        .groupby(
            ["itemid", "label"],
            dropna=False
        )
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )


    print("=" * 70)
    print("3-3-1. Non-continuous Binary Presence Candidates")
    print("=" * 70)

    print(f"Presence candidate ITEMIDs: {presence['itemid'].nunique()}")
    print(f"Presence candidate rows: {len(presence)}")

    print(
        "Missing starttime:",
        presence["presence_starttime"].isna().sum()
    )

    print(
        "Missing endtime:",
        presence["presence_endtime"].isna().sum()
    )


    print("\n[Top 30 ITEMIDs]")

    print(
        summary
        .head(30)
        .to_string(index=False)
    )


    print("\n[Examples]")

    cols = [
        "stay_id",
        "itemid",
        "label",
        "ordercategorydescription",
        "presence",
        "presence_starttime",
        "presence_endtime"
    ]

    print(
        presence[cols]
        .head(20)
        .to_string(index=False)
    )


    print("\n[Important]")
    print("These are presence CANDIDATES, not final pharmaceutical variables.")
    print("Final drug selection and drug-class merging are performed later.")

    print("=" * 70)


    return presence, summary