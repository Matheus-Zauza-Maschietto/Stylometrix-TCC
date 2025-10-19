import pandas as pd

class PandasService:
    @staticmethod
    def clean_non_numeric_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
        for col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors='coerce').fillna(0)
        return metrics