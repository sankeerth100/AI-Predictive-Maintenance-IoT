def get_risk_level(probability):
    if probability < 0.25:
        return "LOW RISK", "Machine is healthy"
    elif probability < 0.50:
        return "MEDIUM RISK", "Monitor machine condition"
    elif probability < 0.75:
        return "HIGH RISK", "Maintenance required soon"
    else:
        return "CRITICAL RISK", "Immediate maintenance required"


def get_recommendation(input_data):
    recommendations = []

    if input_data["Tool wear [min]"] > 180:
        recommendations.append("Tool wear is high. Replace or inspect cutting tool.")

    if input_data["Torque [Nm]"] > 55:
        recommendations.append("Torque is high. Check motor load and shaft alignment.")

    if input_data["Rotational speed [rpm]"] < 1300:
        recommendations.append("Low rotational speed detected. Inspect bearings and motor.")

    if input_data["Air temperature [K]"] > 305:
        recommendations.append("High air temperature. Check cooling and ventilation system.")

    if input_data["Process temperature [K]"] > 315:
        recommendations.append("High process temperature. Inspect thermal control system.")

    if not recommendations:
        recommendations.append("Machine condition is normal. Continue regular monitoring.")

    return recommendations