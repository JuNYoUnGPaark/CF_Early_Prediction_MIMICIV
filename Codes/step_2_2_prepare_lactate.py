import pandas as pd

def prepare_lactate(stays: pd.DataFrame, chartevents: pd.DataFrame, labevents: pd.DataFrame):

    # 현재까지 필터링 통과한 stay만 선택
    current_stay_ids = set(stays["stay_id"])

    stay_windows = stays[["stay_id", "hadm_id", "intime", "outtime"]].copy()
    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"], errors="coerce")
    stay_windows["outtime"] = pd.to_datetime(stay_windows["outtime"], errors="coerce")

    # chartevents의 Lactate 추출
    chart_lactate = chartevents.loc[(chartevents["stay_id"].isin(current_stay_ids)) 
                                    & (chartevents["itemid"] == 225668) 
                                    & chartevents["valuenum"].notna(), ["stay_id", "charttime", "valuenum"]].copy()

    chart_lactate["charttime"] = pd.to_datetime(chart_lactate["charttime"], errors="coerce")
    chart_lactate = chart_lactate.dropna(subset=["charttime"])

    chart_lactate["itemid"] = 225668
    chart_lactate["source_table"] = "chartevents"

    # labevents의 Lactate 추출
    lab_lactate = labevents.loc[(labevents["itemid"] == 50813) & labevents["valuenum"].notna(), ["hadm_id", "charttime", "valuenum"]].copy()

    lab_lactate["charttime"] = pd.to_datetime(lab_lactate["charttime"], errors="coerce")
    lab_lactate = lab_lactate.dropna(subset=["charttime"])

    lab_lactate = lab_lactate.merge(stay_windows, on="hadm_id", how="inner")
    lab_lactate = lab_lactate.loc[(lab_lactate["charttime"] >= lab_lactate["intime"]) & (lab_lactate["charttime"] <= lab_lactate["outtime"])].copy()

    lab_lactate = lab_lactate[["stay_id", "charttime", "valuenum"]]
    lab_lactate["itemid"] = 50813
    lab_lactate["source_table"] = "labevents"


    # 두 Lactate source를 하나의 dataframe으로 concat
    lactate_events = pd.concat([lab_lactate, chart_lactate], ignore_index=True)
    lactate_events = lactate_events.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    # Lactate 기록 보유 stay 확인
    lactate_stay_ids = set(lactate_events["stay_id"].unique())
    # 없는 stay 확인 
    no_lactate_stay_ids = current_stay_ids - lactate_stay_ids


    report = {
        "current_stays": int(len(stays)),
        "labevents_50813_rows": int(len(lab_lactate)),
        "chartevents_225668_rows": int(len(chart_lactate)),
        "combined_lactate_rows": int(len(lactate_events)),
        "stays_with_lactate": int(len(lactate_stay_ids)),
        "stays_without_lactate": int(len(no_lactate_stay_ids))
    }

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