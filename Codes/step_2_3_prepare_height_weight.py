import pandas as pd

def prepare_height_weight(stays: pd.DataFrame, chartevents: pd.DataFrame):

    # 현재 preprocessing 대상 stay만 사용
    current_stay_ids = set(stays["stay_id"])
    
    # Height / Admission Weight 기록 추출
    target_itemids = [226730, 226512, 226531]

    hw_events = chartevents.loc[
        chartevents["stay_id"].isin(current_stay_ids)
        & chartevents["itemid"].isin(target_itemids)
        & chartevents["valuenum"].notna(),
        ["stay_id", "itemid", "charttime", "valuenum"]
    ].copy()

    hw_events["charttime"] = pd.to_datetime(hw_events["charttime"], errors="coerce")
    hw_events = hw_events.dropna(subset=["charttime"]).reset_index(drop=True)

    # 4. 변수 이름 설정해놓기 
    hw_events["variable"] = hw_events["itemid"].map({
        226730: "Height",
        226512: "Weight",
        226531: "Weight"
    })

    #  단위 통일
    # Weight:
    #   226512: kg
    #   226531: ib
    # kg = lbs × 0.453592
    hw_events["value"] = hw_events["valuenum"]
    lbs_mask = hw_events["itemid"] == 226531
    hw_events.loc[lbs_mask, "value"] = hw_events.loc[lbs_mask, "valuenum"] * 0.453592
    hw_events["unit"] = hw_events["variable"].map({
        "Height": "cm",
        "Weight": "kg"
    })

    # 시간순 정렬
    hw_events = hw_events.sort_values(["stay_id", "charttime", "itemid"]).reset_index(drop=True)


    # Height / Weight 분리하여 결과 확인
    height_events = hw_events.loc[hw_events["variable"] == "Height"].copy()
    weight_events = hw_events.loc[hw_events["variable"] == "Weight"].copy()

    height_stays = set(height_events["stay_id"].unique())
    weight_stays = set(weight_events["stay_id"].unique())

    report = {
        "current_stays": int(len(stays)),
        "height_rows": int(len(height_events)),
        "height_stays": int(len(height_stays)),
        "weight_rows": int(len(weight_events)),
        "weight_stays": int(len(weight_stays)),
        "stays_without_height": int(len(current_stay_ids - height_stays)),
        "stays_without_weight": int(len(current_stay_ids - weight_stays))
    }

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