import pandas as pd

# ============================================================
# 3-1. Pharmaceutical Administration Type
# ============================================================
# 원 논문에서는 pharmaceutical variable을
# 최종적으로 rate 또는 presence indicator 형태로 사용함.
#
# Continuous하게 투여되는 약물은 그대로 rate 형태로 사용할 수 있지만,
# bolus / injection / tablet처럼 한 번에 투여되는 약물은
# acting period를 이용해서 일정 시간 동안의 effective continuous rate로 변환함.
#
# 만약 정량적인 rate로 표현하기 어려운 경우에는
# 해당 약물이 현재 존재하는지만 나타내는 binary presence indicator를 사용함.
#
# MIMIC-IV inputevents에는 투여 형태가
# ordercategorydescription으로 구분되어 있어서
# 이를 이용해 continuous / non-continuous를 나눔.
#
#   Continuous Med
#   → 약물을 일정 rate로 계속 주입
#   → continuous
#
#   Continuous IV
#   → IV fluid / 약물을 일정 시간 계속 주입
#   → continuous
#
#   Drug Push
#   → 주사기로 한 번에 밀어 넣는 투여 방식
#   → non-continuous
#
#   Bolus
#   → 한 번에 일정량 투여
#   → non-continuous
#
#   Non IV Meds
#   → 비정맥 약물
#   → non-continuous
#
# 즉 MIMIC-IV에서는 ordercategorydescription을 이용해서
# 먼저 투여 형태를 continuous / non-continuous로 구분하고,
# 이후 각 형태에 맞게 rate / effective rate / presence 형태로 처리함.
# ============================================================

ADMINISTRATION_TYPE_MAP = {
    "Continuous Med": "continuous",
    "Continuous IV": "continuous",

    "Drug Push": "non-continuous",
    "Bolus": "non-continuous",
    "Non Iv Meds": "non-continuous",
}


def classify_inputevents(
    stays: pd.DataFrame,
    inputevents: pd.DataFrame,
    d_items: pd.DataFrame
):

    current_stays = set(
        stays["stay_id"].dropna().unique()
    )

    events = inputevents.loc[
        inputevents["stay_id"].isin(current_stays)
    ].copy()

    events["starttime"] = pd.to_datetime(
        events["starttime"],
        errors="coerce"
    )

    if "endtime" in events.columns:
        events["endtime"] = pd.to_datetime(
            events["endtime"],
            errors="coerce"
        )

    # d_items 정보 붙이기
    item_cols = [
        c for c in ["itemid", "label", "category"]
        if c in d_items.columns
    ]

    if "itemid" in item_cols:

        item_info = (
            d_items[item_cols]
            .drop_duplicates("itemid")
        )

        events = events.merge(
            item_info,
            on="itemid",
            how="left"
        )

    # 5. Continuous / Non-continuous 분류
    events["administration_type"] = (
        events["ordercategorydescription"]
        .map(ADMINISTRATION_TYPE_MAP)
        .fillna("unknown")
    )

    #  전체 category별 report
    category_report = (
        events
        .groupby(
            ["ordercategorydescription", "administration_type"],
            dropna=False
        )
        .agg(
            rows=("itemid", "size"),
            stays=("stay_id", "nunique"),
            unique_itemids=("itemid", "nunique")
        )
        .reset_index()
    )


    # rate / amount availability도 같이 확인
    if "rate" in events.columns:

        rate_report = (
            events
            .assign(rate_present=events["rate"].notna())
            .groupby("administration_type")
            .agg(
                rows=("itemid", "size"),
                rate_present=("rate_present", "sum")
            )
            .reset_index()
        )

        rate_report["rate_missing"] = (
            rate_report["rows"]
            - rate_report["rate_present"]
        )

    else:
        rate_report = None


    if "amount" in events.columns:

        amount_report = (
            events
            .assign(amount_present=events["amount"].notna())
            .groupby("administration_type")
            .agg(
                rows=("itemid", "size"),
                amount_present=("amount_present", "sum")
            )
            .reset_index()
        )

        amount_report["amount_missing"] = (
            amount_report["rows"]
            - amount_report["amount_present"]
        )

    else:
        amount_report = None

    # Administration type별 ITEMID 수
    type_report = (
        events
        .groupby("administration_type")
        .agg(
            rows=("itemid", "size"),
            stays=("stay_id", "nunique"),
            unique_itemids=("itemid", "nunique")
        )
        .reset_index()
    )

    # inputevents가 전혀 없는 ICU stay 확인
    stays_with_inputevents = set(
        events["stay_id"].dropna().unique()
    )

    stays_without_inputevents = (
        current_stays - stays_with_inputevents
    )

    # Unknown category 확인
    unknown_rows = events.loc[
        events["administration_type"] == "unknown"
    ].copy()

    report = {
        "current_stays": int(len(current_stays)),
        "stays_with_inputevents": int(len(stays_with_inputevents)),
        "stays_without_inputevents": int(len(stays_without_inputevents)),

        "inputevent_rows": int(len(events)),

        "continuous_rows": int(
            (events["administration_type"] == "continuous").sum()
        ),

        "noncontinuous_rows": int(
            (events["administration_type"] == "non-continuous").sum()
        ),

        "unknown_rows": int(
            (events["administration_type"] == "unknown").sum()
        ),
    }

    print("=" * 70)
    print("3-1. Inputevents Administration Type Classification")
    print("=" * 70)

    print(f"Current ICU stays: {report['current_stays']}")
    print(f"Stays with inputevents: {report['stays_with_inputevents']}")
    print(f"Stays without inputevents: {report['stays_without_inputevents']}")

    print(f"\nInputevents rows: {report['inputevent_rows']}")

    print("\n[Administration type]")
    print(type_report.to_string(index=False))


    print("\n[Original MIMIC-IV category]")
    print(category_report.to_string(index=False))


    if rate_report is not None:

        print("\n[Rate availability]")
        print(
            rate_report.to_string(index=False)
        )


    if amount_report is not None:

        print("\n[Amount availability]")
        print(
            amount_report.to_string(index=False)
        )


    print("\n[Unknown]")
    print(f"Unknown rows: {len(unknown_rows)}")

    if len(unknown_rows) > 0:

        print(
            unknown_rows[
                [
                    c for c in [
                        "itemid",
                        "label",
                        "ordercategorydescription"
                    ]
                    if c in unknown_rows.columns
                ]
            ]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )


    print("\n[Important]")
    print("No inputevents rows were removed.")
    print("This step only classifies administration form.")
    print("Final pharmaceutical-variable selection is performed later.")

    print("=" * 70)


    return (
        events,
        category_report,
        type_report,
        rate_report,
        amount_report,
        unknown_rows,
        report
    )