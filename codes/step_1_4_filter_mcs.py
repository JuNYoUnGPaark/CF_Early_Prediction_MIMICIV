import pandas as pd


# ============================================================
# 1-4. Mechanical Circulatory Support Exclusion
# ============================================================

def filter_mcs(stays: pd.DataFrame, d_items: pd.DataFrame, chartevents: pd.DataFrame, procedureevents: pd.DataFrame):

    # --------------------------------------------------------
    # 1. 필요한 column 존재 여부 확인
    # --------------------------------------------------------
    # stays:
    #   현재까지 preprocessing을 통과한 ICU stay 목록.
    #   1-3 Age filtering 이후라면 2955 stays가 입력되어야 한다.
    #
    # d_items:
    #   MIMIC-IV ICU variable dictionary.
    #   itemid  = 각 variable의 고유 ID
    #   label   = 해당 variable이 무엇을 의미하는지 나타내는 이름
    #   linksto = 실제 값이 기록되어 있는 MIMIC-IV table
    #
    # chartevents / procedureevents:
    #   d_items에서 찾은 ITEMID가 실제 환자에게 기록됐는지 확인하는 table.
    # --------------------------------------------------------

    if "stay_id" not in stays.columns:
        raise ValueError("stays에 stay_id column이 없습니다.")

    missing = {"itemid", "label", "linksto"} - set(d_items.columns)
    if missing:
        raise ValueError(f"d_items에 필요한 column이 없습니다: {missing}")

    missing = {"stay_id", "itemid"} - set(chartevents.columns)
    if missing:
        raise ValueError(f"chartevents에 필요한 column이 없습니다: {missing}")

    missing = {"stay_id", "itemid"} - set(procedureevents.columns)
    if missing:
        raise ValueError(f"procedureevents에 필요한 column이 없습니다: {missing}")


    # --------------------------------------------------------
    # 2. MCS device 이름 정의
    # --------------------------------------------------------
    # HiRID와 MIMIC-III에서 MCS exclusion에 사용했던 device를
    # MIMIC-IV 구조에 맞게 다음 5개 family로 사용한다.
    #
    #   1. ECMO
    #   2. Impella
    #   3. LVAD
    #   4. RVAD
    #   5. Assist Device
    #
    # MIMIC-IV에는 LVAD/RVAD가 약어가 아니라
    #
    #   Left Ventricular Assit Device
    #   Right Ventricular Assist Device
    #
    # 처럼 기록된 경우도 있으므로 이것들도 함께 검색한다.
    #
    # 주의:
    # MIMIC d_items에는 실제로 "Assit"라고 오타가 있는 label도 있으므로
    # Assist / Assit 둘 다 검색한다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 3. d_items.label에서 MCS 관련 ITEMID 찾기
    # --------------------------------------------------------
    # 여기서는 keyword가 단순히 다른 설명에 우연히 들어간 것이 아니라
    # 해당 장치의 측정/상태/line 등을 나타내는 MIMIC-IV ITEMID를 찾는다.
    #
    # 현재 3000-stay subset에서 실제 사용할 수 있는 table은
    # chartevents와 procedureevents이므로 두 table에 연결되는 ITEMID만 사용한다.
    #
    # 따라서 datetimeevents에 연결된 dressing change / insertion date 등은
    # 현재 구현 대상에서 제외된다.
    #
    # inputevents의 Heparin Sodium (Impella), Bivalirudin (Impella) 등은
    # 약물 투여 기록이지 MCS 장치 자체의 기록이 아니므로 사용하지 않는다.
    # --------------------------------------------------------

    mcs_items = d_items.loc[
        d_items["label"].fillna("").str.contains(pattern, case=False, regex=True)
        & d_items["linksto"].isin(["chartevents", "procedureevents"]),
        ["itemid", "label", "linksto"]
    ].copy()

    mcs_items = mcs_items.drop_duplicates(subset="itemid").reset_index(drop=True)


    # --------------------------------------------------------
    # 4. 각 ITEMID를 5개 device family 중 하나로 분류
    # --------------------------------------------------------
    # 나중에 결과를 확인할 때 단순히 ITEMID 숫자만 보는 것보다
    # 어느 장치 때문에 제외됐는지를 확인하기 위해 family를 붙인다.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 5. 현재 preprocessing 대상 ICU stay만 검사
    # --------------------------------------------------------
    # 여기에는 1-3 Age filtering 결과가 전달되어야 한다.
    #
    # 즉 현재 실행에서는:
    #
    #   원래 3000 stays
    #       ↓ Age filtering
    #   2955 stays
    #       ↓ MCS filtering
    #
    # 순서가 되어야 한다.
    # --------------------------------------------------------

    current_stay_ids = set(stays["stay_id"])


    # --------------------------------------------------------
    # 6. chartevents에서 MCS 관련 실제 기록 찾기
    # --------------------------------------------------------
    # d_items에서 linksto == chartevents인 MCS ITEMID만 가져온다.
    #
    # 그리고:
    #   현재 분석 대상 stay이고
    #   해당 ITEMID가 실제로 기록된 경우
    #
    # 만 선택한다.
    # --------------------------------------------------------

    chart_itemids = set(mcs_items.loc[mcs_items["linksto"] == "chartevents", "itemid"])

    chart_mcs = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin(chart_itemids),
        ["stay_id", "itemid"]
    ].copy()

    chart_mcs["source_table"] = "chartevents"


    # --------------------------------------------------------
    # 7. procedureevents에서 MCS 관련 실제 기록 찾기
    # --------------------------------------------------------
    # ECMO Inflow/Outflow Line, Impella Line 등과 같이
    # procedureevents에 기록된 device 관련 ITEMID를 확인한다.
    # --------------------------------------------------------

    procedure_itemids = set(mcs_items.loc[mcs_items["linksto"] == "procedureevents", "itemid"])

    procedure_mcs = procedureevents.loc[
        procedureevents["stay_id"].isin(current_stay_ids)
        & procedureevents["itemid"].isin(procedure_itemids),
        ["stay_id", "itemid"]
    ].copy()

    procedure_mcs["source_table"] = "procedureevents"


    # --------------------------------------------------------
    # 8. 두 table의 MCS 기록 합치기
    # --------------------------------------------------------

    mcs_events = pd.concat([chart_mcs, procedure_mcs], ignore_index=True)


    # --------------------------------------------------------
    # 9. ITEMID의 label과 device family 연결
    # --------------------------------------------------------
    # 결과 확인 시:
    #
    #   stay_id
    #   itemid
    #   label
    #   device_family
    #   source_table
    #
    # 를 모두 볼 수 있도록 한다.
    # --------------------------------------------------------

    mcs_events = mcs_events.merge(
        mcs_items[["itemid", "label", "device_family"]],
        on="itemid",
        how="left"
    )


    # --------------------------------------------------------
    # 10. MCS가 실제 기록된 ICU stay 찾기
    # --------------------------------------------------------
    # 한 환자에게 같은 ECMO 기록이 수백 번 있어도
    # exclusion에서는 해당 stay를 한 번만 제거하면 된다.
    # --------------------------------------------------------

    mcs_stay_ids = set(mcs_events["stay_id"].dropna().unique())


    # --------------------------------------------------------
    # 11. MCS stay 제외
    # --------------------------------------------------------

    filtered_stays = stays.loc[~stays["stay_id"].isin(mcs_stay_ids)].copy().reset_index(drop=True)


    # --------------------------------------------------------
    # 12. Device family별 MCS stay 수 계산
    # --------------------------------------------------------
    # 같은 stay에서 같은 device 기록이 여러 번 있어도
    # stay-device 조합은 하나로 계산한다.
    #
    # 한 stay가 ECMO와 RVAD를 둘 다 가지고 있다면
    # 각 family count에는 각각 포함될 수 있다.
    #
    # 따라서 family count들의 합이 전체 excluded stay 수보다
    # 클 수 있으며 이는 오류가 아니다.
    # --------------------------------------------------------

    device_counts = (
        mcs_events[["stay_id", "device_family"]]
        .drop_duplicates()
        ["device_family"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # 13. 결과 report
    # --------------------------------------------------------

    report = {
        "before_stays": int(len(stays)),
        "mcs_itemids": int(mcs_items["itemid"].nunique()),
        "mcs_event_rows": int(len(mcs_events)),
        "excluded_stays": int(len(mcs_stay_ids)),
        "after_stays": int(len(filtered_stays)),
        "device_family_stays": device_counts
    }


    # --------------------------------------------------------
    # 14. 결과 출력
    # --------------------------------------------------------

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