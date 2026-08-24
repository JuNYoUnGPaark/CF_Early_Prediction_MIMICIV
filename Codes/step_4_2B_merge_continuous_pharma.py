import pandas as pd


# ============================================================
# 4-2B. Prepare / Merge Continuous Pharmaceutical Events
# ============================================================
#
# 입력:
#   continuous_events  : Step 3-2 결과
#   pharma_map         : Step 4-1B 결과
#
# 목적:
#   1) 최종 pharma ITEMID만 선택
#   2) canonical pharma 이름 붙이기
#   3) 완전히 동일한 infusion record duplicate 제거
#   4) 겹치는 infusion interval이 있는지 audit
#
# 주의:
#   - overlapping infusion의 rate 합산은 아직 하지 않는다.
#   - 실제 시간축 합산은 이후 5-min grid 생성 시 처리한다.
#   - unit harmonization도 아직 하지 않는다.
# ============================================================


def merge_continuous_pharma(
    continuous_events: pd.DataFrame,
    pharma_map: pd.DataFrame,
):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------

    required_cont = {
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "continuous_rate",
        "continuous_rate_uom",
    }

    missing = required_cont - set(continuous_events.columns)

    if missing:
        raise ValueError(
            f"continuous_events에 필요한 column이 없습니다: {missing}"
        )

    required_map = {
        "itemid",
        "pharma_id",
        "pharma_variable",
    }

    missing = required_map - set(pharma_map.columns)

    if missing:
        raise ValueError(
            f"pharma_map에 필요한 column이 없습니다: {missing}"
        )


    # --------------------------------------------------------
    # 2. 최종 pharma ITEMID만 선택 + canonical 이름 연결
    # --------------------------------------------------------

    df = continuous_events.merge(
        pharma_map[
            [
                "itemid",
                "pharma_id",
                "pharma_variable",
            ]
        ],
        on="itemid",
        how="inner"
    ).copy()

    df["starttime"] = pd.to_datetime(
        df["starttime"],
        errors="coerce"
    )

    df["endtime"] = pd.to_datetime(
        df["endtime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "stay_id",
            "starttime",
            "endtime",
            "continuous_rate",
        ]
    ).copy()

    rows_before = len(df)


    # --------------------------------------------------------
    # 3. 완전히 동일한 infusion duplicate 제거
    #
    # 같은 patient / drug / interval / rate / unit이면
    # 같은 기록으로 보고 하나만 유지한다.
    # --------------------------------------------------------

    duplicate_key = [
        "stay_id",
        "pharma_id",
        "starttime",
        "endtime",
        "continuous_rate",
        "continuous_rate_uom",
    ]

    exact_duplicate_rows = int(
        df.duplicated(
            subset=duplicate_key,
            keep="first"
        ).sum()
    )

    df = (
        df.drop_duplicates(
            subset=duplicate_key,
            keep="first"
        )
        .copy()
    )


    # --------------------------------------------------------
    # 4. Interval validity
    # --------------------------------------------------------

    invalid_interval_mask = (
        df["endtime"] <= df["starttime"]
    )

    invalid_interval_rows = int(
        invalid_interval_mask.sum()
    )

    # 잘못된 interval은 이후 시간축에서 사용할 수 없으므로 제거
    df = df.loc[
        ~invalid_interval_mask
    ].copy()


    # --------------------------------------------------------
    # 5. 겹치는 infusion interval audit
    #
    # 같은 stay + pharma 안에서 starttime 순으로 정렬한 뒤,
    # 이전 event의 최대 endtime보다 현재 starttime이 빠르면
    # overlap이 존재한다.
    #
    # 여기서는 "확인"만 하고 rate를 합치지 않는다.
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "stay_id",
            "pharma_id",
            "starttime",
            "endtime",
        ]
    ).reset_index(drop=True)

    previous_max_end = (
        df.groupby(
            ["stay_id", "pharma_id"],
            sort=False
        )["endtime"]
        .cummax()
        .groupby(
            [
                df["stay_id"],
                df["pharma_id"],
            ],
            sort=False
        )
        .shift(1)
    )

    df["overlaps_previous"] = (
        previous_max_end.notna()
        & (df["starttime"] < previous_max_end)
    )

    overlap_rows = int(
        df["overlaps_previous"].sum()
    )


    # --------------------------------------------------------
    # 6. Unit audit
    # --------------------------------------------------------

    unit_report = (
        df.groupby(
            ["pharma_id", "pharma_variable"],
            dropna=False
        )["continuous_rate_uom"]
        .agg(
            unit_count=lambda x: x.dropna().nunique(),
            units=lambda x: " | ".join(
                sorted(set(x.dropna().astype(str)))
            ),
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # 7. Drug-level summary
    # --------------------------------------------------------

    pharma_report = (
        df.groupby(
            ["pharma_id", "pharma_variable"],
            dropna=False
        )
        .agg(
            rows=("itemid", "size"),
            stays=("stay_id", "nunique"),
            source_itemids=("itemid", "nunique"),
            overlap_rows=("overlaps_previous", "sum"),
        )
        .reset_index()
        .sort_values(
            "rows",
            ascending=False
        )
    )


    # --------------------------------------------------------
    # 8. 최종 duplicate check
    # --------------------------------------------------------

    remaining_exact_duplicates = int(
        df.duplicated(
            subset=duplicate_key,
            keep=False
        ).sum()
    )


    # --------------------------------------------------------
    # 9. 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("4-2B. Continuous Pharmaceutical Merge")
    print("=" * 70)

    print(f"Mapped pharma rows before cleanup: {rows_before}")
    print(f"Exact duplicate rows removed: {exact_duplicate_rows}")
    print(f"Invalid interval rows removed: {invalid_interval_rows}")
    print(f"Rows after cleanup: {len(df)}")
    print(
        "Exact duplicate rows remaining:",
        remaining_exact_duplicates
    )
    print(f"Rows overlapping a previous infusion: {overlap_rows}")


    print("\n[Pharmaceutical variables]")
    print(
        pharma_report.to_string(index=False)
    )


    print("\n[Rate units]")
    print(
        unit_report.to_string(index=False)
    )


    print("\n[Important]")
    print(
        "Only exact duplicate infusion records were removed."
    )
    print(
        "Overlapping infusion intervals were preserved."
    )
    print(
        "Overlapping rates will be combined later when constructing "
        "the time grid."
    )
    print(
        "No rate-unit harmonization was performed."
    )

    print("=" * 70)


    report = {
        "rows_before": int(rows_before),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "invalid_interval_rows_removed": invalid_interval_rows,
        "rows_after": int(len(df)),
        "remaining_exact_duplicate_rows": remaining_exact_duplicates,
        "overlap_rows": overlap_rows,
    }


    return (
        df,
        pharma_report,
        unit_report,
        report,
    )