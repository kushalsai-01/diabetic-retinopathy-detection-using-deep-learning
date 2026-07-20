CLINICAL_RECOMMENDATIONS = {
    0: "No Diabetic Retinopathy detected. Continue annual screening and maintain optimal glycemic control (HbA1c < 7%).",
    1: "Mild Non-Proliferative Diabetic Retinopathy detected. Optimize glycemic control. Follow-up in 12 months.",
    2: "Moderate Non-Proliferative Diabetic Retinopathy detected. Refer to ophthalmologist within 3 months.",
    3: "Severe Non-Proliferative Diabetic Retinopathy detected. Urgent ophthalmology referral required within 1 month.",
    4: "Proliferative Diabetic Retinopathy detected. URGENT: Immediate ophthalmology referral required."
}

REFERRAL_URGENCY = {
    0: "routine",
    1: "routine",
    2: "soon",
    3: "urgent",
    4: "emergency",
}

def get_recommendation(grade: int) -> str:
    if grade not in CLINICAL_RECOMMENDATIONS:
        raise ValueError(f"Grade must be 0-4, got {grade}")
    return CLINICAL_RECOMMENDATIONS[grade]

def get_urgency(grade: int) -> str:
    return REFERRAL_URGENCY.get(grade, "unknown")
