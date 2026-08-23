import re
import pandas as pd


# ============================================================
# 3-3A. Build Acting-period Mapping Candidates
# ============================================================
#
# 하는 일:
# 1. 현재 non-continuous ITEMID 목록 생성
# 2. Supplementary Table 4 drugs sheet 읽기
# 3. 약 이름이 정규화 후 정확히 같은 경우만 자동 매칭
# 4. 전체 결과를 CSV로 저장
#
# 하지 않는 일:
# - fuzzy matching
# - synonym 임의 판단
# - acting period 적용
# - effective rate 계산
# ============================================================


def _normalize_name(x):
    """대소문자/공백/기호 차이만 제거."""
    if pd.isna(x):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower()
    )


def build_acting_period_candidates(
    inputevents_classified,
    table4_path,
    output_path
):

    # --------------------------------------------------------
    # 1. 현재 non-continuous ITEMID 목록
    # --------------------------------------------------------

    noncontinuous = inputevents_classified[
        inputevents_classified["administration_type"]
        == "non-continuous"
    ].copy()

    items = (
        noncontinuous
        .groupby(["itemid", "label"], dropna=False)
        .agg(
            rows=("itemid", "size"),
            categories=(
                "ordercategorydescription",
                lambda x: ", ".join(sorted(set(x.dropna())))
            )
        )
        .reset_index()
    )

    items["name_norm"] = items["label"].apply(_normalize_name)


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


    # 필요한 column만 사용
    ref = table4[
        [
            "temp: ID",
            "temp: bern name",
            "constituent drugs (if relevant)",
            "acting period (individual)"
        ]
    ].copy()


    # merged-cell 때문에 비어 있는 group 정보만 forward fill
    ref["temp: ID"] = ref["temp: ID"].ffill()
    ref["temp: bern name"] = ref["temp: bern name"].ffill()


    ref = ref.rename(
        columns={
            "temp: ID": "table4_id",
            "temp: bern name": "table4_group",
            "constituent drugs (if relevant)": "table4_drug",
            "acting period (individual)": "acting_period"
        }
    )


    # acting period가 실제로 정의된 개별 약물만 사용
    ref = ref[
        ref["table4_drug"].notna()
        & ref["acting_period"].notna()
    ].copy()

    ref["name_norm"] = ref["table4_drug"].apply(_normalize_name)


    # --------------------------------------------------------
    # 3. 이름이 정확히 같은 경우만 자동 matching
    # --------------------------------------------------------

    result = items.merge(
        ref[
            [
                "name_norm",
                "table4_id",
                "table4_group",
                "table4_drug",
                "acting_period"
            ]
        ],
        on="name_norm",
        how="left"
    )


    result["mapping_status"] = result["table4_drug"].notna().map(
        {
            True: "exact_name_match",
            False: "unmapped"
        }
    )


    # 보기 편하게 정리
    result = result[
        [
            "itemid",
            "label",
            "rows",
            "categories",
            "table4_id",
            "table4_group",
            "table4_drug",
            "acting_period",
            "mapping_status"
        ]
    ].sort_values(
        ["mapping_status", "rows"],
        ascending=[True, False]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 4. 저장
    # --------------------------------------------------------

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 5. 결과 출력
    # --------------------------------------------------------

    n_itemids = items["itemid"].nunique()

    exact_itemids = result.loc[
        result["mapping_status"] == "exact_name_match",
        "itemid"
    ].nunique()

    unmapped_itemids = result.loc[
        result["mapping_status"] == "unmapped",
        "itemid"
    ].nunique()


    print("=" * 70)
    print("3-3A. Acting-period Mapping Candidates")
    print("=" * 70)

    print(f"Non-continuous ITEMIDs: {n_itemids}")
    print(f"Exact name matches: {exact_itemids}")
    print(f"Unmapped ITEMIDs: {unmapped_itemids}")

    print("\n[Exact matches]")

    exact = result[
        result["mapping_status"] == "exact_name_match"
    ]

    if len(exact):
        print(
            exact[
                [
                    "itemid",
                    "label",
                    "table4_drug",
                    "acting_period"
                ]
            ].to_string(index=False)
        )
    else:
        print("None")

    print("\nSaved:")
    print(output_path)

    print("=" * 70)

    return result, ref