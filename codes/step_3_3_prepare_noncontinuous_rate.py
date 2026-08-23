import pandas as pd


# ============================================================
# 3-3B. Non-continuous Drug → Effective Continuous Rate
# ============================================================
#
# Supplementary Table 4와 이름이 정확히 일치한 약물만 사용한다.
#
# effective rate = amount / acting period
#
# 예:
# 100 mcg Fentanyl, acting period = 1h
# → 100 mcg/hour를 1시간 동안 적용
#
# 기존 inputevents.endtime은 사용하지 않는다.
# 여기서 필요한 기간은 실제 투여시간이 아니라
# Table 4에서 정의한 약물의 acting period이기 때문이다.
# ============================================================


def _acting_period_to_hours(x):

    x = str(x).strip().lower()

    if x.endswith("m"):
        return float(x[:-1]) / 60

    if x.endswith("h"):
        return float(x[:-1])

    if x.endswith("d"):
        return float(x[:-1]) * 24

    raise ValueError(f"Unknown acting period: {x}")


def prepare_noncontinuous_effective_rate(
    inputevents_classified,
    acting_period_candidates
):

    # --------------------------------------------------------
    # 1. 방금 확정한 exact-name match만 사용
    # --------------------------------------------------------

    mapping = acting_period_candidates[
        acting_period_candidates["mapping_status"]
        == "exact_name_match"
    ][
        ["itemid", "table4_drug", "acting_period"]
    ].drop_duplicates("itemid").copy()


    mapping["acting_period_hours"] = (
        mapping["acting_period"]
        .apply(_acting_period_to_hours)
    )


    # --------------------------------------------------------
    # 2. Non-continuous event만 선택
    # --------------------------------------------------------

    events = inputevents_classified[
        inputevents_classified["administration_type"]
        == "non-continuous"
    ].copy()


    # --------------------------------------------------------
    # 3. 17개 mapping이 있는 event만 남기기
    # --------------------------------------------------------

    events = events.merge(
        mapping,
        on="itemid",
        how="inner"
    )


    # --------------------------------------------------------
    # 4. Effective rate 계산
    # --------------------------------------------------------

    events["effective_rate"] = (
        events["amount"]
        / events["acting_period_hours"]
    )

    events["effective_rate_uom"] = (
        events["amountuom"].astype(str)
        + "/hour"
    )


    # --------------------------------------------------------
    # 5. 효과 지속 구간 생성
    # --------------------------------------------------------

    events["effective_starttime"] = pd.to_datetime(
        events["starttime"]
    )

    events["effective_endtime"] = (
        events["effective_starttime"]
        + pd.to_timedelta(
            events["acting_period_hours"],
            unit="h"
        )
    )


    # --------------------------------------------------------
    # 6. 확인
    # --------------------------------------------------------

    print("=" * 70)
    print("3-3B. Non-continuous Effective Rate")
    print("=" * 70)

    print(f"Mapped ITEMIDs: {events['itemid'].nunique()}")
    print(f"Converted rows: {len(events)}")

    print(
        "Missing amount:",
        events["amount"].isna().sum()
    )

    print(
        "Missing effective rate:",
        events["effective_rate"].isna().sum()
    )


    print("\n[Drug summary]")

    summary = (
        events
        .groupby(
            [
                "itemid",
                "label",
                "acting_period"
            ],
            dropna=False
        )
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )

    print(summary.to_string(index=False))


    print("\n[Examples]")

    cols = [
        "stay_id",
        "itemid",
        "label",
        "starttime",
        "amount",
        "amountuom",
        "acting_period",
        "effective_rate",
        "effective_rate_uom",
        "effective_starttime",
        "effective_endtime",
    ]

    print(
        events[cols]
        .head(20)
        .to_string(index=False)
    )


    print("\n[Important]")
    print("Only exact-name matched drugs were converted.")
    print("The remaining non-continuous ITEMIDs were not assigned an acting period.")
    print("=" * 70)


    return events, summary