import pandas as pd

def prepare_noncontinuous_presence(
    inputevents_classified,
    acting_period_candidates
):
    
    # Acting period가 확인된 17개 ITEMID
    quantitative_itemids = set(
        acting_period_candidates.loc[
            acting_period_candidates["mapping_status"] == "exact_name_match",
            "itemid"
        ]
    )

    # Non-continuous events
    events = inputevents_classified[
        inputevents_classified["administration_type"]
        == "non-continuous"
    ].copy()

    # 이미 effective rate로 바꾼 17개는 제외
    presence = events[
        ~events["itemid"].isin(quantitative_itemids)
    ].copy()

    # Binary presence
    presence["presence"] = 1

    presence["presence_starttime"] = pd.to_datetime(
        presence["starttime"],
        errors="coerce"
    )

    presence["presence_endtime"] = pd.to_datetime(
        presence["endtime"],
        errors="coerce"
    )

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