import pandas as pd


# ============================================================
# 3-1. Classify Inputevents by Administration Type
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 원 논문에서는 pharmaceutical variables를 처리할 때
#
#   1) continuous infusion
#   2) bolus / injection / tablet 등의 non-continuous administration
#
# 을 구분하였다.
#
# Continuous:
#   실제 infusion rate를 이용할 수 있음
#
# Non-continuous:
#   이후 drug-specific acting period를 이용하여
#   effective continuous rate로 변환해야 함
#
#
# MIMIC-IV adaptation
# ------------------------------------------------------------
# inputevents.ordercategorydescription을 이용하여
# 투여 형태를 구분한다.
#
# Continuous Med → continuous
# Continuous IV  → continuous
#
# Drug Push      → non-continuous
# Bolus          → non-continuous
# Non Iv Meds    → non-continuous
#
# 그 외          → unknown
#
#
# 주의
# ------------------------------------------------------------
# inputevents에는 약물뿐 아니라
#
#   NaCl
#   Dextrose
#   blood products
#   flush
#   nutrition
#
# 등도 존재한다.
#
# 따라서 이 단계에서는 "최종 pharmaceutical variable"을
# 선택하는 것이 아니라 inputevents의 administration form만
# 분류한다.
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

    # ========================================================
    # 1. 필요한 column 확인
    # ========================================================

    required = {
        "stay_id",
        "itemid",
        "starttime",
        "ordercategorydescription"
    }

    missing = required - set(inputevents.columns)

    if missing:
        raise ValueError(
            f"inputevents에 필요한 column이 없습니다: {missing}"
        )


    # ========================================================
    # 2. 현재 cohort의 ICU stays만 사용
    # ========================================================

    current_stays = set(
        stays["stay_id"].dropna().unique()
    )

    events = inputevents.loc[
        inputevents["stay_id"].isin(current_stays)
    ].copy()


    # ========================================================
    # 3. 시간 column 정리
    # ========================================================

    events["starttime"] = pd.to_datetime(
        events["starttime"],
        errors="coerce"
    )

    if "endtime" in events.columns:
        events["endtime"] = pd.to_datetime(
            events["endtime"],
            errors="coerce"
        )


    # ========================================================
    # 4. d_items 정보 붙이기
    # ========================================================
    #
    # ITEMID만 보면 어떤 입력인지 알기 어렵기 때문에
    # label과 category를 붙여둔다.
    #
    # 이후 실제 pharmaceutical variable mapping에서도
    # 사용할 수 있다.
    # ========================================================

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


    # ========================================================
    # 5. Continuous / Non-continuous 분류
    # ========================================================

    events["administration_type"] = (
        events["ordercategorydescription"]
        .map(ADMINISTRATION_TYPE_MAP)
        .fillna("unknown")
    )


    # ========================================================
    # 6. 전체 category별 report
    # ========================================================

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


    # ========================================================
    # 7. Administration type별 ITEMID 수
    # ========================================================

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


    # ========================================================
    # 8. inputevents가 전혀 없는 ICU stay 확인
    # ========================================================
    #
    # 약물이 없다는 이유로 stay를 제외하지 않는다.
    #
    # 이후 pharmaceutical feature에서는
    # 해당 약물이 존재하지 않는 상태로 처리된다.
    # ========================================================

    stays_with_inputevents = set(
        events["stay_id"].dropna().unique()
    )

    stays_without_inputevents = (
        current_stays - stays_with_inputevents
    )


    # ========================================================
    # 9. Unknown category 확인
    # ========================================================

    unknown_rows = events.loc[
        events["administration_type"] == "unknown"
    ].copy()


    # ========================================================
    # 10. 결과 report
    # ========================================================

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


    # ========================================================
    # 11. 출력
    # ========================================================

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