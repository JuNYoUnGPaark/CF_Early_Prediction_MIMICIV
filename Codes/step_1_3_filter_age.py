import pandas as pd
 
def filter_age(
        stays: pd.DataFrame,
        admissions: pd.DataFrame,
        patients: pd.DataFrame,
):
    
    # 1. 필요한 column들 존재여부 확인
    required_stay_cols = {"stay_id", "subject_id", "hadm_id"}
    required_admission_cols = {"subject_id", "hadm_id", "admittime"}
    requried_patient_cols = {"subject_id", "anchor_age", "anchor_year"}

    missing = required_stay_cols - set(stays.columns)
    if missing: raise ValueError(f"stays에 필요한 col없음: {missing}")
    missing = required_admission_cols - set(admissions.columns)
    if missing: raise ValueError(f"admissions에 필요한 col없음: {missing}")
    missing = requried_patient_cols - set(patients.columns)
    if missing: raise ValueError(f"patients에 필요한 col없음: {missing}")

    # 2. stays에서 필요한 정보만 가져오기
    age_info = stays[["stay_id", "subject_id", "hadm_id"]].copy()

    # 3. admission의 admittime과 age_info 연결하기
    age_info = age_info.merge(admissions[["subject_id", "hadm_id", "admittime"]],
                              on=["subject_id", "hadm_id"],
                              how="left",
                              # 한 admission에서 여러 ICU가 있을 수 있으므로 여러 ICU stay -> 하나의 admission으로 처리
                              validate="many_to_one")   

    # 4. patients에서 anchor_age & anchor_year 연결
    # patients table은 한 subject당 한 행: 여러 ICU stay -> 한 patient 구조 
    age_info = age_info.merge(patients[["subject_id", "anchor_age", "anchor_year",]],
                              on="subject_id",
                              how="left",
                              validate="many_to_one")

    # 5. admittime을 datetime으로 변환
    # admittime에서 year만 추출해야하므로 pandas datetime 형태로 변환
    age_info["admittime"] = pd.to_datetime(age_info["admittime"], errors="coerce")

    # 6. Admission 당시 Age 계산
    age_info["admission_age"] = (age_info["anchor_age"] + (age_info["admittime"].dt.year - age_info["anchor_year"]))

    # 7-1. 89세 초과 환자는 anchor_age=91로 만들어놨고, 이건 제외 
    topcoded_age = age_info["anchor_age"].eq(91)

    # 7-2. age 계산에 필요한 정보가 없는 경우 제외
    missing_age = age_info["admission_age"].isna()

    # 7-3. 16세 미만 제외
    under_16 = (age_info["admission_age"] < 16)

    # 7-4. 100세 초과 제외 (anchor_age=91은 topcoded_age에서 따로 제외)
    over_100 = (age_info["admission_age"] > 100)

    exclude = (missing_age | topcoded_age | under_16 | over_100)

    # 8. 제외 원인 저장
    age_info["exclusion_reason"] = "included" 
    age_info.loc[under_16, "exclusion_reason"] = "age < 16"
    age_info.loc[over_100, "exclusion_reason"] = "age > 100"
    age_info.loc[topcoded_age, "exclusion_reason"] = "anchor_age = 91 (top-coded age)"
    age_info.loc[missing_age, "exclusion_reason"] = "missing age information"

    # 9. 통과된 stay_id만 선택 
    included_stay_ids = age_info.loc[~exclude, "stay_id"]
    filtered_stays = (stays[stays["stay_id"].isin(included_stay_ids)].copy().reset_index(drop=True))

    # 10. 결과 출력
    report = {
        "before_stays": int(len(stays)),

        "missing_age_information": int(missing_age.sum()),

        "anchor_age_91_topcoded": int(topcoded_age.sum()),

        "age_under_16": int(under_16.sum()),

        "age_over_100": int(over_100.sum()),

        "excluded_stays": int(exclude.sum()),

        "after_stays": int(len(filtered_stays)),
    }

    print("=" * 70)
    print("1-3. Age Exclusion")
    print("=" * 70)
    print(f"Before stays: "f"{report['before_stays']}")
    print(f"Missing age information: "f"{report['missing_age_information']}")
    print(f"anchor_age = 91: "f"{report['anchor_age_91_topcoded']}")
    print(f"Age < 16: "f"{report['age_under_16']}")
    print(f"Age > 100: "f"{report['age_over_100']}")
    print(f"Total excluded: "f"{report['excluded_stays']}")
    print(f"After stays: "f"{report['after_stays']}")

    # 분포 확인 
    included_age = age_info.loc[~exclude,"admission_age"]
    if len(included_age) > 0:
        print("\n[Included admission age]")
        print(f"Min: {included_age.min():.1f}")
        print(f"Median: {included_age.median():.1f}")
        print(f"Max: {included_age.max():.1f}")
    print("=" * 70)

    return filtered_stays, age_info, report