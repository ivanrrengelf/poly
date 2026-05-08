"""Inspección rápida del dataset de features."""
import pandas as pd

df = pd.read_csv("data/processed/features.csv")
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df["target"].value_counts())
print(f"\nNaN por columna (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print(f"\nEstadisticas basicas:")
print(df[["price", "momentum_1", "volatility_10", "trade_count_1h", "target"]].describe().to_string())
