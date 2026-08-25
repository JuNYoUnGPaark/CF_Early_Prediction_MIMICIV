import pandas as pd

def filter_mcs(stays: pd.DataFrame, d_items: pd.DataFrame, chartevents: pd.DataFrame, procedureevents: pd.DataFrame):
    # MCS device 이름 정의
    pattern = (
        r"ECMO|"
        r"Impella|"
        r"LVAD|"
        r"RVAD|"
        r"Assist Device|"
        r"Assit Device|"
        r"Ventricular Assist Device|"
        r"Ventricular Assit Device"
    )

    # d_items.label에서 MCS 관련 ITEMID 찾기
    mcs_items = d_items.loc[
        d_items["label"].fillna("").str.contains(pattern, case=False, regex=True)
        & d_items["linksto"].isin(["chartevents", "procedureevents"]),
        ["itemid", "label", "linksto"]
    ].copy()
    mcs_items = mcs_items.drop_duplicates(subset="itemid").reset_index(drop=True)

    # 각 ITEMID를 5개 device family 중 하나로 분류
    def get_device_family(label):
        label = str(label).lower()
        if "ecmo" in label:
            return "ECMO"
        if "impella" in label:
            return "Impella"
        if "lvad" in label or "left ventricular assist" in label or "left ventricular assit" in label:
            return "LVAD"
        if "rvad" in label or "right ventricular assist" in label or "right ventricular assit" in label:
            return "RVAD"
        if "assist device" in label or "assit device" in label:
            return "Assist Device"
        return "Unknown"
    mcs_items["device_family"] = mcs_items["label"].apply(get_device_family)

    # 현재까지 필터링 통과한 ICU stay만 검사
    current_stay_ids = set(stays["stay_id"])

    # chartevents에서 MCS 관련 실제 기록 찾기
    chart_itemids = set(mcs_items.loc[mcs_items["linksto"] == "chartevents", "itemid"])

    # 필터링 통과 stay면서 chartevents에 itemid가 있는 stay_id, itemid 가져오기 
    chart_mcs = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin(chart_itemids),
        ["stay_id", "itemid"]
    ].copy()
    chart_mcs["source_table"] = "chartevents"


    # procedureevents에서도 MCS 관련 실제 기록 찾기
    procedure_itemids = set(mcs_items.loc[mcs_items["linksto"] == "procedureevents", "itemid"])
    procedure_mcs = procedureevents.loc[
        procedureevents["stay_id"].isin(current_stay_ids)
        & procedureevents["itemid"].isin(procedure_itemids),
        ["stay_id", "itemid"]
    ].copy()
    procedure_mcs["source_table"] = "procedureevents"

    # 두 table MCS 기록 합치기
    mcs_events = pd.concat([chart_mcs, procedure_mcs], ignore_index=True)

    # ITEMID의 label과 device family 연결
    mcs_events = mcs_events.merge(
        mcs_items[["itemid", "label", "device_family"]],
        on="itemid",
        how="left"
    )

    # MCS가 실제 기록된 ICU stay 찾기
    mcs_stay_ids = set(mcs_events["stay_id"].dropna().unique())
    
    # MCS stay 제외
    filtered_stays = stays.loc[~stays["stay_id"].isin(mcs_stay_ids)].copy().reset_index(drop=True)

    # Device family별 MCS stay 수 계산
    device_counts = (
        mcs_events[["stay_id", "device_family"]]
        .drop_duplicates()
        ["device_family"]
        .value_counts()
        .to_dict()
    )

    report = {
        "before_stays": int(len(stays)),
        "mcs_itemids": int(mcs_items["itemid"].nunique()),
        "mcs_event_rows": int(len(mcs_events)),
        "excluded_stays": int(len(mcs_stay_ids)),
        "after_stays": int(len(filtered_stays)),
        "device_family_stays": device_counts
    }

    print("=" * 70)
    print("1-4. Mechanical Circulatory Support Exclusion")
    print("=" * 70)

    print(f"Before stays: {report['before_stays']}")
    print(f"MCS ITEMIDs found: {report['mcs_itemids']}")
    print(f"MCS event rows: {report['mcs_event_rows']}")
    print(f"MCS stays excluded: {report['excluded_stays']}")
    print(f"After stays: {report['after_stays']}")


    # 실제 사용된 ITEMID 확인
    print("\n[MCS ITEMIDs]")
    print(mcs_items.to_string(index=False))


    # device별 실제 stay 수 확인
    print("\n[MCS stays by device]")

    if device_counts:
        print(pd.Series(device_counts))
    else:
        print("No MCS stay found.")


    # 제외된 stay 일부 확인
    print("\n[Excluded MCS stay samples]")

    if len(mcs_events) > 0:
        print(
            mcs_events[["stay_id", "itemid", "label", "device_family", "source_table"]]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )
    else:
        print("No MCS event found.")


    print("=" * 70)

    return filtered_stays, mcs_items, mcs_events, report