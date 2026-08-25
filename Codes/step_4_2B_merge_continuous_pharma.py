import pandas as pd

def merge_continuous_pharma(
    continuous_events: pd.DataFrame,
    pharma_map: pd.DataFrame,
):
    # 최종 pharma ITEMID만 선택 + canonical 이름 연결
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

    # 완전히 동일한 infusion duplicate 제거
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

    # Interval validity
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

    remaining_exact_duplicates = int(
        df.duplicated(
            subset=duplicate_key,
            keep=False
        ).sum()
    )

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