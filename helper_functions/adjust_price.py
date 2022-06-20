import pandas as pd
import numpy as np
from decimal import ROUND_HALF_UP, Decimal

# Adjust Close Price according to adjustment factor

def adjust_price(data = pd.read_csv("modified_data/stock_prices.csv")):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    data.loc[: ,"Date"] = pd.to_datetime(data.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        df.loc[:, "AdjustedOpen"] = (
            df["CumulativeAdjustmentFactor"] * df["Open"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        df.loc[:, "AdjustedHigh"] = (
            df["CumulativeAdjustmentFactor"] * df["High"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        df.loc[:, "AdjustedLow"] = (
            df["CumulativeAdjustmentFactor"] * df["Low"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df.loc[df["AdjustedOpen"] == 0, "AdjustedOpen"] = np.nan
        df.loc[df["AdjustedHigh"] == 0, "AdjustedHigh"] = np.nan
        df.loc[df["AdjustedLow"] == 0, "AdjustedLow"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        df.loc[:, "AdjustedOpen"] = df.loc[:, "AdjustedOpen"].ffill()
        df.loc[:, "AdjustedHigh"] = df.loc[:, "AdjustedHigh"].ffill()
        df.loc[:, "AdjustedLow"] = df.loc[:, "AdjustedLow"].ffill()
        return df

    # generate AdjustedClose
    price = data.sort_values(["SecuritiesCode", "Date"])
    price = data.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    del price["Open"]
    del price["High"]
    del price["Low"]
    del price["Close"]
    price.to_csv("modified_data/stock_prices_adjusted.csv")
    return price