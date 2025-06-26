def interpret_prediction(prob, shap_values_row, features_row):
    explanation = []

    # Confidence comment
    if 0.45 <= prob <= 0.55:
        explanation.append("🟡 Model is uncertain — the customer is near the decision boundary.")
    elif prob > 0.75:
        explanation.append("🔴 Model is very confident that the customer is likely to churn.")
    elif prob < 0.25:
        explanation.append("🟢 Model is very confident that the customer is not likely to churn.")

    # SHAP-based reasoning
    sorted_indices = abs(shap_values_row).argsort()[::-1]
    for i in sorted_indices[:3]:  # Top 3 reasons
        feature = features_row.index[i]
        value = features_row.iloc[i]
        shap_val = shap_values_row[i]
        direction = "↑" if shap_val > 0 else "↓"
        impact = "increases" if shap_val > 0 else "decreases"

        # Custom description (you can refine per feature)
        if feature == "InternetService_Fiber optic":
            explanation.append(f"- Uses fiber optic internet ({direction}), which {impact} churn risk")
        elif feature == "tenure":
            explanation.append(f"- Has been a customer for {value} months ({direction})")
        elif feature == "charges_ratio":
            explanation.append(f"- Charges ratio of {round(value,2)} ({direction})")
        elif feature == "total_services":
            explanation.append(f"- Subscribes to {value} services ({direction})")
        elif feature == "StreamingTV_Yes":
            explanation.append(f"- Streams TV content ({direction})")
        elif feature == "StreamingMovies_Yes":
            explanation.append(f"- Streams movies ({direction})")
        elif feature == "is_multiple_services":
            explanation.append(f"- Uses multiple services ({direction})")
        elif feature == "TechSupport_No internet service":
            explanation.append(f"- No internet for tech support ({direction})")
        elif feature == "MultipleLines_Yes":
            explanation.append(f"- Has multiple lines ({direction})")
        elif feature == "tenure_group_49-72":
            explanation.append(f"- In tenure group 49-72 months ({direction})")
        else:
            explanation.append(f"- {feature}: {value} ({direction})")

    return "\n".join(explanation)
