import pandas as pd
import numpy as np
import argparse

def df_statistics(X: pd.DataFrame):
    arr = X.values
    means = arr.mean(axis=0)
    vars = arr.var(axis=0)

    return (means, vars)
    
def generate(args):
    df_path = args.path_df
    path_to_write = args.path_to
    df = pd.read_csv(df_path)
    means, vars = df_statistics(df)
    samples = np.random.normal(means, vars, df.shape)
    pd.DataFrame(samples).to_csv(path_to_write+"/synthetic_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_df", type=str)
    parser.add_argument("path_to", type=str)
    args = parser.parse_args()
    generate(args)

