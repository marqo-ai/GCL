import pandas as pd
import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", default="/MarqoModels/GE/marqo-ecommerce-B")
    parser.add_argument("--metrics", default="['MAP@1000', 'P@10', 'Recall@10', 'MRR@1000', 'NDCG@10']")
    parser.add_argument("--evals", default="['gs-title2image', 'gs-query2image', 'ap-name2image']")

    args = parser.parse_args()

    evals = eval(args.evals)

    metrics = eval(args.metrics)
    exp_path = args.exp_path
    exp_dirs = [os.path.join(exp_path, n) for n in evals]

    all_results = {"name": []}
    all_results.update({m: [] for m in metrics})
    for exp_path in exp_dirs:
        with open(os.path.join(exp_path, "output.json"), "r") as f:
            results = json.load(f)
        del results['summary']
        results = {k: v for d in results.values() for k, v in d.items() if k in all_results}
        results["name"] = os.path.basename(exp_path)
        for k in all_results:
            all_results[k].append(results[k])

    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(exp_path, "results_sum.csv"), index=False)
    os.system(f"cat {os.path.join(exp_path, 'results_sum.csv')}")
