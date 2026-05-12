import os

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://orchestration:4200/api")

print("🔄 Retraining aangevraagd via Prefect API...")
print(f"   Prefect API: {PREFECT_API_URL}")
print("   In productie zou dit een Prefect deployment triggeren.")
print("   Voor nu: logt de retraining aanvraag.")
