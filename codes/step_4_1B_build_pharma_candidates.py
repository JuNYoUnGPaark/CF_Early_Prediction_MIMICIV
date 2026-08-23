import re
import pandas as pd


# ============================================================
# 4-1B. Build Pharmaceutical Target Mapping Candidates
# ============================================================
#
# 목적:
# 현재 MIMIC-IV inputevents의 ITEMID가
# Supplementary Table 4의 어떤 pharmaceutical drug/group과
# 정확히 대응되는지 확인한다.
#
# 이 단계에서는:
# - fuzzy matching 안 함
# - synonym 수동 mapping 안 함
# - 실제 merging 안 함
#
# 이름이 정규화 후 정확히 같은 경우만 후보로 만든다.
# ============================================================


def _normalize_name(x):
    if pd.isna(x):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower()
    )


def build_pharma_target_candidates(
    inputevents_classified,
    table4_path,
    output_path
):

    # --------------------------------------------------------
    # 1. 현재 MIMIC-IV input ITEMID 요약
    # --------------------------------------------------------

    items = (
        inputevents_classified
        .groupby(["itemid", "label"], dropna=False)
        .agg(
            rows=("itemid", "size"),

            continuous_rows=(
                "administration_type",
                lambda x: (x == "continuous").sum()
            ),

            noncontinuous_rows=(
                "administration_type",
                lambda x: (x == "non-continuous").sum()
            ),
        )
        .reset_index()
    )

    items["name_norm"] = items["label"].apply(
        _normalize_name
    )


    # --------------------------------------------------------
    # 2. Supplementary Table 4 drugs sheet
    # --------------------------------------------------------

    table4 = pd.read_excel(
        table4_path,
        sheet_name="drugs"
    )

    table4.columns = [
        str(c).strip()
        for c in table4.columns
    ]


    ref = table4[
        [
            "temp: ID",
            "temp: bern name",
            "constituent drugs (if relevant)"
        ]
    ].copy()


    # Excel merged cell 복원
    ref["temp: ID"] = ref["temp: ID"].ffill()
    ref["temp: bern name"] = ref["temp: bern name"].ffill()


    ref = ref.rename(
        columns={
            "temp: ID": "table4_id",
            "temp: bern name": "table4_group",
            "constituent drugs (if relevant)": "table4_drug"
        }
    )


    # --------------------------------------------------------
    # 3. constituent drug exact match
    # --------------------------------------------------------

    constituent_ref = ref[
        ref["table4_drug"].notna()
    ].copy()

    constituent_ref["name_norm"] = (
        constituent_ref["table4_drug"]
        .apply(_normalize_name)
    )

    constituent_match = items.merge(
        constituent_ref[
            [
                "name_norm",
                "table4_id",
                "table4_group",
                "table4_drug"
            ]
        ],
        on="name_norm",
        how="inner"
    )

    constituent_match["match_type"] = \
        "exact_constituent"


    # --------------------------------------------------------
    # 4. Table4 group name 자체와 exact match
    # --------------------------------------------------------

    group_ref = (
        ref[
            ["table4_id", "table4_group"]
        ]
        .dropna()
        .drop_duplicates()
    )

    group_ref["name_norm"] = (
        group_ref["table4_group"]
        .apply(_normalize_name)
    )

    group_match = items.merge(
        group_ref,
        on="name_norm",
        how="inner"
    )

    group_match["table4_drug"] = None
    group_match["match_type"] = "exact_group"


    # --------------------------------------------------------
    # 5. 두 종류 match 합치기
    # --------------------------------------------------------

    matches = pd.concat(
        [
            constituent_match,
            group_match
        ],
        ignore_index=True
    )

    matches = matches[
        [
            "itemid",
            "label",
            "rows",
            "continuous_rows",
            "noncontinuous_rows",
            "table4_id",
            "table4_group",
            "table4_drug",
            "match_type"
        ]
    ].drop_duplicates()


    # --------------------------------------------------------
    # 6. 같은 ITEMID가 여러 Table4 target에 연결되는지 확인
    # --------------------------------------------------------

    target_count = (
        matches
        .groupby("itemid")["table4_group"]
        .nunique()
        .rename("n_table4_targets")
        .reset_index()
    )

    matches = matches.merge(
        target_count,
        on="itemid",
        how="left"
    )

    matches["mapping_status"] = matches[
        "n_table4_targets"
    ].map(
        lambda n:
        "exact_unique"
        if n == 1
        else "ambiguous"
    )


    matches = matches.sort_values(
        ["mapping_status", "rows"],
        ascending=[True, False]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 7. 저장
    # --------------------------------------------------------

    matches.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 8. 출력
    # --------------------------------------------------------

    unique_exact = matches.loc[
        matches["mapping_status"] == "exact_unique",
        "itemid"
    ].nunique()

    ambiguous = matches.loc[
        matches["mapping_status"] == "ambiguous",
        "itemid"
    ].nunique()


    print("=" * 70)
    print("4-1B. Pharmaceutical Target Mapping Candidates")
    print("=" * 70)

    print(
        f"Current input ITEMIDs: "
        f"{items['itemid'].nunique()}"
    )

    print(
        f"Exact unique Table4 matches: "
        f"{unique_exact}"
    )

    print(
        f"Ambiguous matches: "
        f"{ambiguous}"
    )


    print("\n[Exact unique matches]")

    safe = matches[
        matches["mapping_status"] == "exact_unique"
    ]

    print(
        safe[
            [
                "itemid",
                "label",
                "continuous_rows",
                "noncontinuous_rows",
                "table4_group",
                "table4_drug",
                "match_type"
            ]
        ].to_string(index=False)
    )


    print("\n[Ambiguous]")

    amb = matches[
        matches["mapping_status"] == "ambiguous"
    ]

    if len(amb):
        print(amb.to_string(index=False))
    else:
        print("None")


    print("\nSaved:")
    print(output_path)

    print("\n[Important]")
    print("No fuzzy or synonym mapping was performed.")
    print("No pharmaceutical events were removed or merged yet.")

    print("=" * 70)


    return matches, items