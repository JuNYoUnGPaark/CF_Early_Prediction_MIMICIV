import numpy as np
import pandas as pd


# ============================================================
# 3-2. Prepare Continuous Administration Rate
# ============================================================
#
# 원 논문:
#   Continuous pharmaceutical variables → rate로 표현
#
# MIMIC-IV adaptation:
#
#   1) inputevents.rate가 존재
#      → 실제 기록된 rate를 그대로 사용
#
#   2) continuous인데 rate가 없음
#      → amount / 투여시간(hour)으로 평균 rate 계산
#
# 주의:
#   기존 rate가 있는 경우에는 절대로 amount/time으로
#   다시 계산하지 않는다.
#
#   기존 rate는 mcg/kg/min 등 환자 체중이 반영된 단위일 수 있기
#   때문에 inputevents의 원래 값을 보존해야 한다.
#
# 단위 통합은 여기서 하지 않는다.
# 이후 Variable Mapping / Merging 단계에서 처리한다.
# ============================================================


def prepare_continuous_rate(inputevents_classified: pd.DataFrame):

    continuous = inputevents_classified.loc[
        inputevents_classified["administration_type"] == "continuous"
    ].copy()

    continuous["starttime"] = pd.to_datetime(
        continuous["starttime"],
        errors="coerce"
    )

    continuous["endtime"] = pd.to_datetime(
        continuous["endtime"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 1. 실제 투여시간 계산
    # --------------------------------------------------------

    continuous["duration_hours"] = (
        continuous["endtime"] - continuous["starttime"]
    ).dt.total_seconds() / 3600.0


    # --------------------------------------------------------
    # 2. 최종 continuous rate column 생성
    # --------------------------------------------------------
    #
    # 기존 rate가 있으면 그대로 사용한다.
    # --------------------------------------------------------

    continuous["continuous_rate"] = continuous["rate"]

    continuous["continuous_rate_uom"] = continuous["rateuom"]

    continuous["rate_source"] = np.where(
        continuous["rate"].notna(),
        "recorded",
        "missing"
    )


    # --------------------------------------------------------
    # 3. rate missing인 경우
    #    amount / duration(hour)
    # --------------------------------------------------------

    reconstruct_mask = (
        continuous["rate"].isna()
        & continuous["amount"].notna()
        & continuous["duration_hours"].notna()
        & (continuous["duration_hours"] > 0)
    )

    continuous.loc[
        reconstruct_mask,
        "continuous_rate"
    ] = (
        continuous.loc[reconstruct_mask, "amount"]
        / continuous.loc[reconstruct_mask, "duration_hours"]
    )


    # --------------------------------------------------------
    # 4. 새 rate의 단위 생성
    # --------------------------------------------------------
    #
    # 예:
    # mEq   → mEq/hour
    # grams → grams/hour
    # mmol  → mmol/hour
    # mg    → mg/hour
    #
    # 기존 rate의 단위는 건드리지 않는다.
    # --------------------------------------------------------

    continuous.loc[
        reconstruct_mask,
        "continuous_rate_uom"
    ] = (
        continuous.loc[reconstruct_mask, "amountuom"]
        .astype(str)
        + "/hour"
    )

    continuous.loc[
        reconstruct_mask,
        "rate_source"
    ] = "derived_amount_duration"


    # --------------------------------------------------------
    # 5. 그래도 rate를 만들지 못한 row 확인
    # --------------------------------------------------------

    unresolved = continuous.loc[
        continuous["continuous_rate"].isna()
    ].copy()


    # --------------------------------------------------------
    # 6. Report
    # --------------------------------------------------------

    recorded_n = int(
        (continuous["rate_source"] == "recorded").sum()
    )

    derived_n = int(
        (
            continuous["rate_source"]
            == "derived_amount_duration"
        ).sum()
    )

    unresolved_n = int(len(unresolved))


    print("=" * 70)
    print("3-2. Continuous Rate Preparation")
    print("=" * 70)

    print(f"Continuous rows: {len(continuous)}")

    print("\n[Rate source]")
    print(f"Existing recorded rate: {recorded_n}")
    print(f"Derived from amount / duration: {derived_n}")
    print(f"Unresolved: {unresolved_n}")


    print("\n[Derived rate units]")

    derived_units = (
        continuous.loc[
            continuous["rate_source"] == "derived_amount_duration",
            "continuous_rate_uom"
        ]
        .value_counts()
    )

    print(derived_units.to_string())


    print("\n[Example derived rates]")

    example_cols = [
        "stay_id",
        "itemid",
        "label",
        "starttime",
        "endtime",
        "amount",
        "amountuom",
        "duration_hours",
        "continuous_rate",
        "continuous_rate_uom",
    ]

    print(
        continuous.loc[
            continuous["rate_source"] == "derived_amount_duration",
            [c for c in example_cols if c in continuous.columns]
        ]
        .head(20)
        .to_string(index=False)
    )


    print("\n[Important]")
    print("Existing inputevents.rate values were preserved.")
    print("Only missing continuous rates were reconstructed.")
    print("No unit harmonization or pharmaceutical variable merging was performed.")

    print("=" * 70)


    return continuous, unresolved