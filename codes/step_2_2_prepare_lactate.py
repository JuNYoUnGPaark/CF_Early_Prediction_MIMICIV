import pandas as pd


# ============================================================
# 2-2. Blood Gas Artifact / Lactate Preparation
# ============================================================

def prepare_lactate(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------
    # MIMIC-IV에서 사용할 Lactate:
    #
    #   chartevents ITEMID 225668 = Lactic Acid
    #   labevents   ITEMID 50813  = Lactate
    #
    # chartevents는 stay_id가 직접 존재하지만,
    # labevents에는 stay_id가 없으므로 hadm_id와 charttime을 이용해
    # 해당 lab measurement가 어느 ICU stay 안에서 측정됐는지 연결한다.
    # --------------------------------------------------------

    required_stays = {"stay_id", "hadm_id", "intime", "outtime"}
    required_chart = {"stay_id", "itemid", "charttime", "valuenum"}
    required_lab = {"hadm_id", "itemid", "charttime", "valuenum"}

    if required_stays - set(stays.columns):
        raise ValueError(f"stays에 필요한 column이 없습니다: {required_stays - set(stays.columns)}")

    if required_chart - set(chartevents.columns):
        raise ValueError(f"chartevents에 필요한 column이 없습니다: {required_chart - set(chartevents.columns)}")

    if required_lab - set(labevents.columns):
        raise ValueError(f"labevents에 필요한 column이 없습니다: {required_lab - set(labevents.columns)}")


    # --------------------------------------------------------
    # 2. 현재 preprocessing 대상 stay 준비
    # --------------------------------------------------------
    # 현재 1-3, 1-4, 1-5를 거친 2878 stays가 입력되어야 한다.
    # --------------------------------------------------------

    current_stay_ids = set(stays["stay_id"])

    stay_windows = stays[["stay_id", "hadm_id", "intime", "outtime"]].copy()
    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"], errors="coerce")
    stay_windows["outtime"] = pd.to_datetime(stay_windows["outtime"], errors="coerce")


    # --------------------------------------------------------
    # 3. chartevents의 Lactate 추출
    # --------------------------------------------------------
    # ITEMID 225668 = Lactic Acid
    #
    # 현재 cohort의 stay이면서 실제 numerical value가 존재하는
    # measurement만 사용한다.
    # --------------------------------------------------------

    chart_lactate = chartevents.loc[(chartevents["stay_id"].isin(current_stay_ids)) & (chartevents["itemid"] == 225668) & chartevents["valuenum"].notna(), ["stay_id", "charttime", "valuenum"]].copy()

    chart_lactate["charttime"] = pd.to_datetime(chart_lactate["charttime"], errors="coerce")
    chart_lactate = chart_lactate.dropna(subset=["charttime"])

    chart_lactate["itemid"] = 225668
    chart_lactate["source_table"] = "chartevents"


    # --------------------------------------------------------
    # 4. labevents의 Lactate 추출
    # --------------------------------------------------------
    # ITEMID 50813 = Lactate
    #
    # labevents에는 stay_id가 없기 때문에:
    #
    #   같은 hadm_id
    #   AND
    #   ICU intime <= charttime <= ICU outtime
    #
    # 조건으로 ICU stay에 연결한다.
    # --------------------------------------------------------

    lab_lactate = labevents.loc[(labevents["itemid"] == 50813) & labevents["valuenum"].notna(), ["hadm_id", "charttime", "valuenum"]].copy()

    lab_lactate["charttime"] = pd.to_datetime(lab_lactate["charttime"], errors="coerce")
    lab_lactate = lab_lactate.dropna(subset=["charttime"])

    lab_lactate = lab_lactate.merge(stay_windows, on="hadm_id", how="inner")
    lab_lactate = lab_lactate.loc[(lab_lactate["charttime"] >= lab_lactate["intime"]) & (lab_lactate["charttime"] <= lab_lactate["outtime"])].copy()

    lab_lactate = lab_lactate[["stay_id", "charttime", "valuenum"]]
    lab_lactate["itemid"] = 50813
    lab_lactate["source_table"] = "labevents"


    # --------------------------------------------------------
    # 5. 두 Lactate source를 하나의 dataframe으로 모으기
    # --------------------------------------------------------
    # 여기서는 단순히 같은 medical concept인 Lactate measurement를
    # 한 곳에 모아두기만 한다.
    #
    # 같은 timestamp에 두 source의 값이 동시에 존재하는 경우의 median
    # 처리는 뒤의 Variable Merging 단계에서 수행한다.
    # --------------------------------------------------------

    lactate_events = pd.concat([lab_lactate, chart_lactate], ignore_index=True)
    lactate_events = lactate_events.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


    # --------------------------------------------------------
    # 6. Lactate 기록 보유 stay 확인
    # --------------------------------------------------------
    # 여기서 Lactate가 없는 stay를 제외하지 않는다.
    #
    # Lactate가 특정 시간 구간에서 없으면 나중 State Annotation에서
    # 해당 5-min time point가 Ambiguous가 될 수 있다.
    # --------------------------------------------------------

    lactate_stay_ids = set(lactate_events["stay_id"].unique())
    no_lactate_stay_ids = current_stay_ids - lactate_stay_ids


    # --------------------------------------------------------
    # 7. 결과 report
    # --------------------------------------------------------

    report = {
        "current_stays": int(len(stays)),
        "labevents_50813_rows": int(len(lab_lactate)),
        "chartevents_225668_rows": int(len(chart_lactate)),
        "combined_lactate_rows": int(len(lactate_events)),
        "stays_with_lactate": int(len(lactate_stay_ids)),
        "stays_without_lactate": int(len(no_lactate_stay_ids))
    }


    # --------------------------------------------------------
    # 8. 결과 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("2-2. Blood Gas Artifact / Lactate Preparation")
    print("=" * 70)

    print("HiRID arterial/venous relabeling: Not applied to MIMIC-IV")
    print(f"Current stays: {report['current_stays']}")
    print(f"Lactate rows - labevents 50813: {report['labevents_50813_rows']}")
    print(f"Lactate rows - chartevents 225668: {report['chartevents_225668_rows']}")
    print(f"Combined Lactate rows: {report['combined_lactate_rows']}")
    print(f"Stays with Lactate: {report['stays_with_lactate']}")
    print(f"Stays without Lactate: {report['stays_without_lactate']}")

    if len(lactate_events) > 0:
        print("\n[Lactate value]")
        print(f"Min: {lactate_events['valuenum'].min()}")
        print(f"Median: {lactate_events['valuenum'].median()}")
        print(f"Max: {lactate_events['valuenum'].max()}")

    print("=" * 70)

    return lactate_events, report