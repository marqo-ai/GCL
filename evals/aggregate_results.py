import pandas as pd
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", default=None)
    parser.add_argument("--evals", default=None)

    args = parser.parse_args()

    evals = eval(args.evals)

    metrics = ["mAP@1000", "mrr@1000", "NDCG@10", "mERR", "mRBP9"]
    exp_path = args.exp_path
    dirs = os.listdir(exp_path)

    all_results = []
    for dir in dirs:
        if dir in evals:
            df = pd.read_csv(os.path.join(exp_path, dir, "output.csv"))
            df["name"] = [dir]
            all_results.append(df)
    all_results = pd.concat(all_results)
    all_results = all_results[["name"] + metrics]
    all_results = all_results.sort_values("name")
    all_results.to_csv(os.path.join(exp_path, "results_sum.csv"), index=False)
    os.system(f"cat {os.path.join(exp_path, 'results_sum.csv')}")
