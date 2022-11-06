import sweetviz as sv
import pandas as pd
import argparse

def make_report(path: str):
    df = pd.read_csv(path)
    sweet_report = sv.analyze(df, target_feat = "condition")
    sweet_report.show_html('EDA_report.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="display a square of a given number",
                    type=str)
    args = parser.parse_args()
    make_report(args.path)
