import pandas as pd

def add_block_aware_rolling_features(df, block_col, window=250):
    """
    Recommended-correction helper.

    Use only when a real session/recording identifier has been established.
    It prevents a rolling history from crossing the supplied block boundary.
    """
    out=df.copy()
    phys=["Diastolic Blood Pressure","Systolic Blood Pressure","Heart Rate","Body Temperature","SPO2"]
    for feature in phys:
        shifted=out.groupby(block_col)[feature].shift(1)
        out[f"{feature}_roll{window}_mean"]=(
            shifted.groupby(out[block_col]).rolling(window,min_periods=window)
            .mean().reset_index(level=0,drop=True)
        )
        out[f"{feature}_roll{window}_std"]=(
            shifted.groupby(out[block_col]).rolling(window,min_periods=window)
            .std().reset_index(level=0,drop=True)
        )
    return out
