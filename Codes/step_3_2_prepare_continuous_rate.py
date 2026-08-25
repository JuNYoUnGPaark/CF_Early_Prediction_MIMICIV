import numpy as np
import pandas as pd

# ============================================================
# 3-2. Continuous Pharmaceutical Rate Preparation
# ============================================================
# 원 논문에서는 모든 pharmaceutical variable을
# 최종적으로 rate 또는 presence indicator 형태로 표현함.
#
# MIMIC-IV inputevents에는 continuous medication에 대해
# 이미 rate column이 존재하므로 이 값을 그대로 활용할 수 있음.
#
# 현재 continuous event 전체:
#   445,889 rows
#
#   기존 rate가 있는 경우
#   → 416,414 rows (93.4%)
#   → inputevents.rate를 그대로 사용
#
#   기존 rate가 없는 경우
#   → 29,475 rows (6.6%)
#   → amount와 투여 duration을 이용해서 rate를 직접 계산
#
# 계산:
#   duration_hours = endtime - starttime
#   rate = amount / duration_hours
#
# 결과적으로 continuous event 445,889개 모두
# 정량적인 rate 형태로 표현할 수 있었음.
#
# 즉 MIMIC-IV에서는 continuous pharmaceutical variable에 대해
# 기존 rate가 있으면 그대로 사용하고,
# 없으면 amount / duration으로 보완해서
# 원 논문의 "rate 형태로 표현"하는 과정을 Adapt함.
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

    # 실제 투여시간 계산
    continuous["duration_hours"] = (
        continuous["endtime"] - continuous["starttime"]
    ).dt.total_seconds() / 3600.0

    # 최종 continuous rate column 생성
    continuous["continuous_rate"] = continuous["rate"]

    continuous["continuous_rate_uom"] = continuous["rateuom"]

    continuous["rate_source"] = np.where(
        continuous["rate"].notna(),
        "recorded",
        "missing"
    )

    # rate missing인 경우 -> amount / duration(hour)
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

    # 새 rate의 단위 생성
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

    # 그래도 rate를 만들지 못한 row 확인
    unresolved = continuous.loc[
        continuous["continuous_rate"].isna()
    ].copy()

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