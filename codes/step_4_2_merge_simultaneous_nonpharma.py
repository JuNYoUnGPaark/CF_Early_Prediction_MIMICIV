import pandas as pd


# ============================================================
# 4-2A. Simultaneous Physiology / Lab Measurements
# ============================================================
#
# 동일 stay + 동일 variable + 동일 timestamp에
# 여러 measurement가 존재하면 median을 사용한다.
#
# 원 논문의 variable merging 이후 duplicate 처리 방식과 동일.
# ============================================================


def merge_simultaneous_nonpharma(nonpharma_events):

    before = len(nonpharma_events)

    # 같은 stay / variable / time에 몇 개가 겹치는지 확인
    group_sizes = (
        nonpharma_events
        .groupby(
            ["stay_id", "variable", "charttime"]
        )
        .size()
        .reset_index(name="n")
    )

    duplicate_groups = int(
        (group_sizes["n"] > 1).sum()
    )


    # --------------------------------------------------------
    # 같은 시각의 값 → median
    # --------------------------------------------------------

    merged = (
        nonpharma_events
        .groupby(
            ["stay_id", "variable", "charttime"],
            as_index=False
        )
        .agg(
            valuenum=("valuenum", "median")
        )
    )

    after = len(merged)


    print("=" * 70)
    print("4-2A. Simultaneous Physiology / Lab Measurements")
    print("=" * 70)

    print(f"Rows before: {before}")
    print(f"Simultaneous groups (>1 row): {duplicate_groups}")
    print(f"Rows after median merging: {after}")
    print(f"Rows reduced: {before - after}")

    print("\n[Variable summary]")

    print(
        merged
        .groupby("variable")
        .agg(
            rows=("valuenum", "size"),
            stays=("stay_id", "nunique")
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .to_string(index=False)
    )

    print("=" * 70)

    return merged