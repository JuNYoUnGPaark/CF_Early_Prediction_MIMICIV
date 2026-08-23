import pandas as pd


# ============================================================
# 1-5. Exclude Stays Without Data for CF Determination
# ============================================================

def filter_cf_data(stays: pd.DataFrame, chartevents: pd.DataFrame):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------
    # 원 저자 코드에서는 이후 State Annotation을 위한 5분 grid를
    # 첫 Heart Rate 측정 ~ 마지막 Heart Rate 측정 구간으로 생성한다.
    #
    # 따라서 Heart Rate 기록이 하나도 없는 patient/stay는
    # 이후 processing 자체가 불가능하므로 제외한다.
    #
    # MIMIC-IV Heart Rate ITEMID:
    #   220045 = Heart Rate
    # --------------------------------------------------------

    if "stay_id" not in stays.columns:
        raise ValueError("stays에 stay_id column이 없습니다.")

    required_cols = {"stay_id", "itemid", "valuenum"}
    missing = required_cols - set(chartevents.columns)

    if missing:
        raise ValueError(f"chartevents에 필요한 column이 없습니다: {missing}")


    # --------------------------------------------------------
    # 2. 현재 preprocessing 대상 stay만 선택
    # --------------------------------------------------------
    # 현재 순서:
    #
    # 3000
    #   ↓ Age exclusion
    # 2955
    #   ↓ MCS exclusion
    # 2878
    #
    # 따라서 여기에는 현재 2878 stays가 입력되어야 한다.
    # --------------------------------------------------------

    current_stay_ids = set(stays["stay_id"])


    # --------------------------------------------------------
    # 3. 실제 Heart Rate 측정값 찾기
    # --------------------------------------------------------
    # 단순히 ITEMID record가 존재하는지만 보는 것이 아니라
    # 실제 numerical measurement가 존재해야 하므로
    # valuenum이 NaN인 record는 사용하지 않는다.
    # --------------------------------------------------------

    hr = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & (chartevents["itemid"] == 220045)
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "valuenum"]
    ].copy()


    # --------------------------------------------------------
    # 4. Heart Rate 기록이 있는 stay 찾기
    # --------------------------------------------------------

    hr_stay_ids = set(hr["stay_id"].unique())


    # --------------------------------------------------------
    # 5. Heart Rate가 하나도 없는 stay 찾기
    # --------------------------------------------------------

    no_hr_stay_ids = current_stay_ids - hr_stay_ids


    # --------------------------------------------------------
    # 6. HR이 없는 stay 제외
    # --------------------------------------------------------

    filtered_stays = stays.loc[~stays["stay_id"].isin(no_hr_stay_ids)].copy().reset_index(drop=True)


    # --------------------------------------------------------
    # 7. stay별 HR 측정 횟수 저장
    # --------------------------------------------------------
    # 단순히 몇 명 제외됐는지만 보는 것보다
    # 각 stay에 HR이 실제로 얼마나 기록됐는지 확인할 수 있도록 한다.
    # --------------------------------------------------------

    hr_counts = hr.groupby("stay_id").size().rename("hr_measurements")

    cf_data_info = stays[["stay_id"]].copy()
    cf_data_info = cf_data_info.merge(hr_counts, on="stay_id", how="left")
    cf_data_info["hr_measurements"] = cf_data_info["hr_measurements"].fillna(0).astype(int)
    cf_data_info["has_hr"] = cf_data_info["hr_measurements"] > 0
    cf_data_info["exclusion_reason"] = "included"
    cf_data_info.loc[~cf_data_info["has_hr"], "exclusion_reason"] = "no Heart Rate data"


    # --------------------------------------------------------
    # 8. 결과 report
    # --------------------------------------------------------

    report = {
        "before_stays": int(len(stays)),
        "stays_with_hr": int(len(hr_stay_ids)),
        "stays_without_hr": int(len(no_hr_stay_ids)),
        "excluded_stays": int(len(no_hr_stay_ids)),
        "after_stays": int(len(filtered_stays))
    }


    # --------------------------------------------------------
    # 9. 결과 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("1-5. CF Data Availability")
    print("=" * 70)

    print(f"Before stays: {report['before_stays']}")
    print(f"Stays with HR data: {report['stays_with_hr']}")
    print(f"Stays without HR data: {report['stays_without_hr']}")
    print(f"Total excluded: {report['excluded_stays']}")
    print(f"After stays: {report['after_stays']}")

    if len(filtered_stays) > 0:
        included_counts = cf_data_info.loc[cf_data_info["has_hr"], "hr_measurements"]

        print("\n[Heart Rate measurements per included stay]")
        print(f"Min: {included_counts.min()}")
        print(f"Median: {included_counts.median():.1f}")
        print(f"Max: {included_counts.max()}")

    if no_hr_stay_ids:
        print("\n[Excluded stays]")
        print(cf_data_info.loc[~cf_data_info["has_hr"]].to_string(index=False))

    print("=" * 70)

    return filtered_stays, cf_data_info, report