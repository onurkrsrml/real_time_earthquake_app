def magnitude_to_score(magnitude):
    if magnitude < 4.0: return 0.20
    elif magnitude < 5.0: return 0.45
    elif magnitude < 6.0: return 0.70
    else: return 0.95

def days_to_urgency_score(days):
    if days <= 3: return 0.95
    elif days <= 7: return 0.80
    elif days <= 14: return 0.60
    elif days <= 30: return 0.35
    else: return 0.15

def calculate_consistency(risk_score, magnitude_score, urgency_score, confidence_score):
    
    diff_1 = abs(risk_score - magnitude_score)
    diff_2 = abs(risk_score - urgency_score)
    
    base_consistency = 1 - ((diff_1 + diff_2) / 2)
    final_consistency = base_consistency * confidence_score
    
    return round(final_consistency, 3)

def consistency_engine(combined_data):
    # Ham verileri al
    risk = combined_data.get("risk_score", 0)
    mag = combined_data.get("predicted_max_magnitude", 0)
    days = combined_data.get("days_to_event", 0)
    conf = combined_data.get("confidence_score", 0)

    # skor dönüşümleri 
    m_score = magnitude_to_score(mag)
    u_score = days_to_urgency_score(days)
    
    cons_score = calculate_consistency(risk, m_score, u_score, conf)

    # Mantıksal Kurallar
    status = "Normal"
    is_consistent = True
    if risk > 0.70 and mag < 4.0:
        status = "Orta/Düşük Tutarlılık"
    elif risk > 0.70 and days <= 7:
        status = "Yüksek Tutarlılık"
    elif risk < 0.40 and mag > 5.5:
        status = "Model Çelişkisi Var"
        is_consistent = False

    # Paketleme
    combined_data["consistency_check"] = {
        "is_consistent": is_consistent,
        "status": status,
        "consistency_score": cons_score 
    }
    return combined_data