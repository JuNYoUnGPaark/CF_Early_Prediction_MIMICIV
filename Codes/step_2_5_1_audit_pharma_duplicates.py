import pandas as pd
import numpy as np


# ============================================================
# 2-5-1. Pharmaceutical Duplicate Audit
# ============================================================
#
# 목적
# ------------------------------------------------------------
# HiRID에서는 pharmaceutical duplicate를 다음처럼 처리했다.
#
#   1) zero dose                     → delete
#   2) tablet / injection duplicate → sum
#   3) other duplicate              → mean
#
# 하지만 MIMIC-IV inputevents에서는 같은 시각의 같은 ITEMID라도
# 서로 다른 orderid / endtime / amount / rate를 가질 수 있다.
#
# 따라서 여기서는 아직 아무 row도 제거하거나 합치지 않는다.
#
# 먼저 MIMIC-IV에서 "duplicate처럼 보이는 record"들이
# 실제로 어떤 구조인지 확인한 뒤 최종 adaptation rule을 정한다.
# ============================================================


def audit_pharma_duplicates(
    stays: pd.DataFrame,
    inputevents: pd.DataFrame,
    d_items: pd.DataFrame
):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------

    required = {"stay_id", "itemid", "starttime"}

    missing = required - set(inputevents.columns)

    if missing:
        raise ValueError(
            f"inputevents에 필요한 column이 없습니다: {missing}"
        )


    # --------------------------------------------------------
    # 2. 현재 2878-stay cohort만 사용
    # --------------------------------------------------------
    #
    # 이전 patient filtering 결과와 동일한 cohort를 유지한다.
    # --------------------------------------------------------

    current_stays = set(stays["stay_id"].dropna().unique())

    ie = inputevents.loc[
        inputevents["stay_id"].isin(current_stays)
    ].copy()

    ie["starttime"] = pd.to_datetime(
        ie["starttime"],
        errors="coerce"
    )

    if "endtime" in ie.columns:
        ie["endtime"] = pd.to_datetime(
            ie["endtime"],
            errors="coerce"
        )

    ie = ie.loc[
        ie["starttime"].notna()
    ].copy()


    # --------------------------------------------------------
    # 3. d_items label 붙이기
    # --------------------------------------------------------
    #
    # duplicate가 많이 발생한 ITEMID가 실제로 어떤 약물/입력인지
    # 사람이 확인하기 쉽게 label을 붙인다.
    # --------------------------------------------------------

    if {"itemid", "label"}.issubset(d_items.columns):

        item_labels = (
            d_items[["itemid", "label"]]
            .drop_duplicates("itemid")
        )

        ie = ie.merge(
            item_labels,
            on="itemid",
            how="left"
        )

    else:
        ie["label"] = np.nan


    # ========================================================
    # 4. Zero-dose 구조 확인
    # ========================================================
    #
    # HiRID의 "zero dose"를 MIMIC-IV amount=0이라고
    # 바로 동일시하지는 않는다.
    #
    # 우선 실제 amount/rate 0 기록이 얼마나 있는지만 확인한다.
    # ========================================================

    zero_amount_rows = 0
    zero_rate_rows = 0

    if "amount" in ie.columns:
        zero_amount_rows = int(
            (ie["amount"].notna() & (ie["amount"] == 0)).sum()
        )

    if "rate" in ie.columns:
        zero_rate_rows = int(
            (ie["rate"].notna() & (ie["rate"] == 0)).sum()
        )


    # ========================================================
    # 5. Duplicate candidate 정의
    # ========================================================
    #
    # MIMIC-IV에서 가장 기본적인 후보:
    #
    #   same stay_id
    #   + same itemid
    #   + same starttime
    #
    # 아직 이것을 "진짜 duplicate"라고 확정하지 않는다.
    #
    # 예:
    #
    # 같은 norepinephrine이 10:00에 시작했지만
    # orderid가 다르고 rate도 다르면
    # 서로 다른 infusion/order일 수도 있다.
    # ========================================================

    group_cols = [
        "stay_id",
        "itemid",
        "starttime"
    ]


    # --------------------------------------------------------
    # 사용할 수 있는 column에 대해서만 nunique 계산
    # --------------------------------------------------------

    agg_dict = {
        "n_rows": ("itemid", "size")
    }

    candidate_columns = [
        "orderid",
        "endtime",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "ordercategorydescription",
        "ordercategoryname",
        "statusdescription"
    ]

    for col in candidate_columns:

        if col in ie.columns:
            agg_dict[f"n_{col}"] = (
                col,
                lambda x: x.nunique(dropna=False)
            )


    group_stats = (
        ie
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )


    # same stay + itemid + starttime에 2개 이상인 경우만
    duplicate_groups = group_stats.loc[
        group_stats["n_rows"] > 1
    ].copy()


    # ========================================================
    # 6. duplicate candidate의 구조 분류
    # ========================================================

    def is_same(df, column):
        name = f"n_{column}"

        if name not in df.columns:
            return pd.Series(False, index=df.index)

        return df[name] == 1


    same_order = is_same(
        duplicate_groups,
        "orderid"
    )

    same_endtime = is_same(
        duplicate_groups,
        "endtime"
    )

    same_amount = is_same(
        duplicate_groups,
        "amount"
    )

    same_rate = is_same(
        duplicate_groups,
        "rate"
    )


    # --------------------------------------------------------
    # A. order / time / amount / rate가 모두 같음
    #
    # → 실제 duplicated record일 가능성이 가장 높은 group
    # --------------------------------------------------------

    duplicate_groups["same_order_same_values"] = (
        same_order
        & same_endtime
        & same_amount
        & same_rate
    )


    # --------------------------------------------------------
    # B. orderid가 다름
    #
    # → 단순 DB duplicate가 아니라
    #   서로 다른 medication order일 가능성
    # --------------------------------------------------------

    if "n_orderid" in duplicate_groups.columns:

        duplicate_groups["different_orderid"] = (
            duplicate_groups["n_orderid"] > 1
        )

    else:

        duplicate_groups["different_orderid"] = False


    # --------------------------------------------------------
    # C. 같은 order인데 amount/rate 등이 다름
    #
    # → order 내부 수정/변경/기록 구조일 가능성
    # --------------------------------------------------------

    duplicate_groups["same_order_different_values"] = (
        same_order
        & (
            ~same_endtime
            | ~same_amount
            | ~same_rate
        )
    )


    # ========================================================
    # 7. Candidate duplicate 원본 rows 추출
    # ========================================================

    duplicate_keys = duplicate_groups[
        group_cols
    ].copy()

    duplicate_keys["_duplicate_candidate"] = True

    duplicate_rows = ie.merge(
        duplicate_keys,
        on=group_cols,
        how="inner"
    )


    # ========================================================
    # 8. ITEMID별 duplicate 통계
    # ========================================================

    item_report = (
        duplicate_groups
        .groupby("itemid")
        .agg(
            duplicate_groups=("n_rows", "size"),
            rows_in_groups=("n_rows", "sum"),
            same_order_same_values=(
                "same_order_same_values",
                "sum"
            ),
            different_orderid=(
                "different_orderid",
                "sum"
            ),
            same_order_different_values=(
                "same_order_different_values",
                "sum"
            )
        )
        .reset_index()
    )


    # label 추가
    labels = (
        ie[["itemid", "label"]]
        .drop_duplicates("itemid")
    )

    item_report = item_report.merge(
        labels,
        on="itemid",
        how="left"
    )

    item_report = item_report.sort_values(
        "duplicate_groups",
        ascending=False
    ).reset_index(drop=True)


    # ========================================================
    # 9. 투여 category 분포
    # ========================================================

    category_report = None

    if "ordercategorydescription" in duplicate_rows.columns:

        category_report = (
            duplicate_rows[
                "ordercategorydescription"
            ]
            .fillna("<missing>")
            .value_counts()
            .rename_axis(
                "ordercategorydescription"
            )
            .reset_index(name="rows")
        )


    # ========================================================
    # 10. Status 분포
    # ========================================================

    status_report = None

    if "statusdescription" in duplicate_rows.columns:

        status_report = (
            duplicate_rows[
                "statusdescription"
            ]
            .fillna("<missing>")
            .value_counts()
            .rename_axis(
                "statusdescription"
            )
            .reset_index(name="rows")
        )


    # ========================================================
    # 11. 전체 report
    # ========================================================

    report = {
        "current_stays": int(
            ie["stay_id"].nunique()
        ),

        "inputevent_rows": int(
            len(ie)
        ),

        "duplicate_groups": int(
            len(duplicate_groups)
        ),

        "rows_in_duplicate_groups": int(
            duplicate_groups["n_rows"].sum()
        ),

        "same_order_same_values": int(
            duplicate_groups[
                "same_order_same_values"
            ].sum()
        ),

        "different_orderid": int(
            duplicate_groups[
                "different_orderid"
            ].sum()
        ),

        "same_order_different_values": int(
            duplicate_groups[
                "same_order_different_values"
            ].sum()
        ),

        "zero_amount_rows": zero_amount_rows,
        "zero_rate_rows": zero_rate_rows,
    }


    # ========================================================
    # 12. 출력
    # ========================================================

    print("=" * 70)
    print("2-5-1. Pharmaceutical Duplicate Audit")
    print("=" * 70)

    print(f"Current stays: {report['current_stays']}")
    print(f"Inputevents rows: {report['inputevent_rows']}")

    print("\n[Same stay + ITEMID + starttime]")
    print(f"Duplicate candidate groups: {report['duplicate_groups']}")
    print(f"Rows in these groups: {report['rows_in_duplicate_groups']}")

    print("\n[Duplicate structure]")
    print(
        "Same order + same endtime/amount/rate: "
        f"{report['same_order_same_values']}"
    )

    print(
        "Different orderid: "
        f"{report['different_orderid']}"
    )

    print(
        "Same order but different endtime/amount/rate: "
        f"{report['same_order_different_values']}"
    )

    print("\n[Zero values]")
    print(
        f"amount == 0 rows: "
        f"{report['zero_amount_rows']}"
    )

    print(
        f"rate == 0 rows: "
        f"{report['zero_rate_rows']}"
    )


    print("\n[Top ITEMIDs with duplicate candidates]")

    if len(item_report) > 0:
        print(
            item_report
            .head(20)
            .to_string(index=False)
        )
    else:
        print("None")


    if category_report is not None:

        print("\n[ordercategorydescription in duplicate rows]")

        print(
            category_report
            .head(20)
            .to_string(index=False)
        )


    if status_report is not None:

        print("\n[statusdescription in duplicate rows]")

        print(
            status_report
            .head(20)
            .to_string(index=False)
        )


    print("\n[Important]")
    print("No inputevents rows were modified.")
    print("same stay + ITEMID + starttime is only a duplicate CANDIDATE.")
    print("Different orderid may represent a legitimate separate medication order.")

    print("=" * 70)


    return (
        duplicate_groups,
        duplicate_rows,
        item_report,
        category_report,
        status_report,
        report
    )