from src.metrics.loader import load_metrics, get_case_metrics

csv_path = "data/per_patient_metrics.csv"
patient_id = "1HNA008"   # change if needed

df = load_metrics(csv_path)
metrics = get_case_metrics(df, patient_id)

print("Metrics loaded:")
for k, v in metrics.items():
    print(f"{k}: {v}")
