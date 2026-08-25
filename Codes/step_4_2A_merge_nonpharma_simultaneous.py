import pandas as pd

def merge_nonpharma_simultaneous(
    nonpharma_events: pd.DataFrame,
):

    required = {
        "stay_id",
        "charttime",
        "variable",
        "valuenum",
        "is_cf_map_source",
    }

    missing = required - set(nonpharma_events.columns)

    if missing:
        raise ValueError(
            f"nonpharma_events에 필요한 column이 없습니다: {missing}"
        )

    df = nonpharma_events.copy()

    df["charttime"] = pd.to_datetime(
        df["charttime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "stay_id",
            "charttime",
            "variable",
            "valuenum",
        ]
    ).copy()

    rows_before = len(df)

    key_cols = [
        "stay_id",
        "charttime",
        "variable",
    ]


    # 1. non-pharma merge
    merged = (
        df.groupby(
            key_cols,
            as_index=False,
            sort=False,
        )["valuenum"]
        .median()
    )


    # 2. CF annotation용 MAP
    merged["cf_map_valuenum"] = merged["valuenum"]

    abp_cf = (
        df[
            (df["variable"] == "ABP mean")
            & (df["is_cf_map_source"] == True)
        ]
        .groupby(
            key_cols,
            as_index=False,
            sort=False,
        )["valuenum"]
        .median()
        .rename(
            columns={
                "valuenum": "abp_cf_valuenum"
            }
        )
    )

    merged = merged.merge(
        abp_cf,
        on=key_cols,
        how="left",
    )

    abp_mask = (
        merged["variable"] == "ABP mean"
    )

    merged.loc[
        abp_mask,
        "cf_map_valuenum"
    ] = merged.loc[
        abp_mask,
        "abp_cf_valuenum"
    ]

    merged = merged.drop(
        columns=["abp_cf_valuenum"]
    )

    rows_after = len(merged)

    remaining_duplicates = int(
        merged.duplicated(
            subset=key_cols
        ).sum()
    )

    report = {
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "rows_removed_by_merge": int(
            rows_before - rows_after
        ),
        "post_duplicate_rows": remaining_duplicates,
    }


    print("=" * 70)
    print("4-2A. Simultaneous Non-pharmaceutical Merge [FAST]")
    print("=" * 70)

    print(f"Rows before merge: {rows_before}")
    print(f"Rows after merge: {rows_after}")
    print(
        "Rows removed by canonical median merge:",
        rows_before - rows_after,
    )
    print(
        "Duplicate rows remaining after merge:",
        remaining_duplicates,
    )

    print("\n[Important]")
    print(
        "Same stay + charttime + canonical variable "
        "values were merged by median."
    )
    print(
        "ABP mean cf_map_valuenum excludes "
        "IABP source ITEMID 224322."
    )

    print("=" * 70)

    return (
        merged,
        report,
    )