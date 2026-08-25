import pandas as pd

def filter_cf_data(stays: pd.DataFrame, chartevents: pd.DataFrame):

    # 현재까지 필터링 통과한 stay만 선택
    current_stay_ids = set(stays["stay_id"])

    # 실제 Heart Rate 측정값 찾기
    hr = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & (chartevents["itemid"] == 220045)
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "valuenum"]
    ].copy()

    # Heart Rate 기록이 있는 stay 찾기
    hr_stay_ids = set(hr["stay_id"].unique())

    # HR이 하나도 없는 기록 찾기 
    no_hr_stay_ids = current_stay_ids - hr_stay_ids

    #  HR이 없는 stay 제외
    filtered_stays = stays.loc[~stays["stay_id"].isin(no_hr_stay_ids)].copy().reset_index(drop=True)

    # stay별 HR 측정 횟수 저장
    hr_counts = hr.groupby("stay_id").size().rename("hr_measurements")

    cf_data_info = stays[["stay_id"]].copy()
    cf_data_info = cf_data_info.merge(hr_counts, on="stay_id", how="left")
    cf_data_info["hr_measurements"] = cf_data_info["hr_measurements"].fillna(0).astype(int)
    cf_data_info["has_hr"] = cf_data_info["hr_measurements"] > 0
    cf_data_info["exclusion_reason"] = "included"
    cf_data_info.loc[~cf_data_info["has_hr"], "exclusion_reason"] = "no Heart Rate data"

    report = {
        "before_stays": int(len(stays)),
        "stays_with_hr": int(len(hr_stay_ids)),
        "stays_without_hr": int(len(no_hr_stay_ids)),
        "excluded_stays": int(len(no_hr_stay_ids)),
        "after_stays": int(len(filtered_stays))
    }

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