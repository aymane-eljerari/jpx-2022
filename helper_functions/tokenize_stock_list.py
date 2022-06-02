import pandas as pd

def tokenize_stock_list(data="data/stock_list.csv"):

    df = pd.read_csv(data)
    
    columns_of_interest = ["Section/Products", "NewMarketSegment"]

    section_set = list(set(df["Section/Products"].tolist()))
    market_set  = list(set(df["NewMarketSegment"].tolist()))

    for i in range(len(df)):
        # +1 to offset the labels
        section_value = df["Section/Products"][i]
        df["Section/Products"][i] = section_set.index(section_value) + 1

        market_value = df["NewMarketSegment"][i]
        df["NewMarketSegment"][i] = market_set.index(market_value) + 1
    
    df.to_csv("modified_data/tokenized_stock_list.csv")

