import pandas as pd


# ============================================================
# 2-3. Height / Weight Artifact Preparation
# ============================================================

def prepare_height_weight(stays: pd.DataFrame, chartevents: pd.DataFrame):

    # --------------------------------------------------------
    # 1. 필요한 column 확인
    # --------------------------------------------------------
    # MIMIC-IV에서 사용할 ITEMID:
    #
    #   226730 = Height (cm)
    #   226512 = Admission Weight (kg)
    #   226531 = Admission Weight (lbs)
    #
    # MIMIC-IV에서는 Height와 Weight가 ITEMID로 명확하게 분리되어 있으므로
    # HiRID처럼 Height와 Weight 값을 서로 swap하지 않는다.
    # --------------------------------------------------------

    if "stay_id" not in stays.columns:
        raise ValueError("stays에 stay_id column이 없습니다.")

    required_cols = {"stay_id", "itemid", "charttime", "valuenum"}
    missing = required_cols - set(chartevents.columns)

    if missing:
        raise ValueError(f"chartevents에 필요한 column이 없습니다: {missing}")


    # --------------------------------------------------------
    # 2. 현재 preprocessing 대상 stay만 사용
    # --------------------------------------------------------
    # 현재:
    #   3000
    #   → Age exclusion
    #   → MCS exclusion
    #   → CF-data availability
    #   = 2878 stays
    # --------------------------------------------------------

    current_stay_ids = set(stays["stay_id"])


    # --------------------------------------------------------
    # 3. Height / Admission Weight 기록 추출
    # --------------------------------------------------------

    target_itemids = [226730, 226512, 226531]

    hw_events = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin(target_itemids)
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "charttime", "valuenum"]
    ].copy()

    hw_events["charttime"] = pd.to_datetime(hw_events["charttime"], errors="coerce")
    hw_events = hw_events.dropna(subset=["charttime"]).reset_index(drop=True)


    # --------------------------------------------------------
    # 4. 변수 이름 부여
    # --------------------------------------------------------

    hw_events["variable"] = hw_events["itemid"].map({
        226730: "Height",
        226512: "Weight",
        226531: "Weight"
    })


    # --------------------------------------------------------
    # 5. 단위 통일
    # --------------------------------------------------------
    # Height:
    #   226730 → cm 그대로 사용
    #
    # Weight:
    #   226512 → kg 그대로 사용
    #   226531 → lbs이므로 kg로 변환
    #
    # MIMIC 저자 코드에서 사용한 변환:
    #   kg = lbs × 0.453592
    # --------------------------------------------------------

    hw_events["value"] = hw_events["valuenum"]

    lbs_mask = hw_events["itemid"] == 226531
    hw_events.loc[lbs_mask, "value"] = hw_events.loc[lbs_mask, "valuenum"] * 0.453592

    hw_events["unit"] = hw_events["variable"].map({
        "Height": "cm",
        "Weight": "kg"
    })


    # --------------------------------------------------------
    # 6. 시간순 정렬
    # --------------------------------------------------------
    # 아직 첫 번째 Height를 static으로 확정하지 않는다.
    #
    # 다음 2-4에서 permitted range를 적용하여 이상치를 제거한 뒤
    # 남아있는 첫 번째 valid Height를 static Height로 사용할 예정이다.
    # --------------------------------------------------------

    hw_events = hw_events.sort_values(["stay_id", "charttime", "itemid"]).reset_index(drop=True)


    # --------------------------------------------------------
    # 7. Height / Weight 분리하여 결과 확인
    # --------------------------------------------------------

    height_events = hw_events.loc[hw_events["variable"] == "Height"].copy()
    weight_events = hw_events.loc[hw_events["variable"] == "Weight"].copy()

    height_stays = set(height_events["stay_id"].unique())
    weight_stays = set(weight_events["stay_id"].unique())


    # --------------------------------------------------------
    # 8. 결과 report
    # --------------------------------------------------------

    report = {
        "current_stays": int(len(stays)),
        "height_rows": int(len(height_events)),
        "height_stays": int(len(height_stays)),
        "weight_rows": int(len(weight_events)),
        "weight_stays": int(len(weight_stays)),
        "stays_without_height": int(len(current_stay_ids - height_stays)),
        "stays_without_weight": int(len(current_stay_ids - weight_stays))
    }


    # --------------------------------------------------------
    # 9. 결과 출력
    # --------------------------------------------------------

    print("=" * 70)
    print("2-3. Height / Weight Artifact Preparation")
    print("=" * 70)

    print("HiRID Height/Weight swap: Not applied to MIMIC-IV")
    print("Weight unit conversion: lbs → kg")
    print(f"Current stays: {report['current_stays']}")

    print(f"\nHeight rows: {report['height_rows']}")
    print(f"Stays with Height: {report['height_stays']}")
    print(f"Stays without Height: {report['stays_without_height']}")

    print(f"\nWeight rows: {report['weight_rows']}")
    print(f"Stays with Weight: {report['weight_stays']}")
    print(f"Stays without Weight: {report['stays_without_weight']}")

    if len(height_events) > 0:
        print("\n[Raw Height (cm)]")
        print(f"Min: {height_events['value'].min()}")
        print(f"Median: {height_events['value'].median()}")
        print(f"Max: {height_events['value'].max()}")

    if len(weight_events) > 0:
        print("\n[Raw Weight (kg, after unit conversion)]")
        print(f"Min: {weight_events['value'].min()}")
        print(f"Median: {weight_events['value'].median()}")
        print(f"Max: {weight_events['value'].max()}")

    print("=" * 70)

    return hw_events, report