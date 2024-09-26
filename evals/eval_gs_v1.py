import sys
import time

import torch
import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
import open_clip
from beir.retrieval.evaluation import EvaluateRetrieval
import logging
from eval_dataset_loader import MFRightEvalDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from pathlib import Path
import numpy
import faiss
from datasets import load_dataset
import zipfile
from transformers import AutoModel, AutoTokenizer


def scan_model_configs():
    _MODEL_CONFIG_PATHS = [Path(__file__).parent / f"cfg/"]
    _MODEL_CONFIGS = {}
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
    from open_clip.factory import _natural_key
    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

    return _MODEL_CONFIGS


def process_multi_modal_args(args):
    args.left_keys = eval(args.left_keys)
    args.right_keys = eval(args.right_keys)
    args.left_weights = eval(args.left_weights)
    args.right_weights = eval(args.right_weights)
    args.img_or_txt = eval(args.img_or_txt)
    assert len(args.left_weights) == len(args.left_keys)
    assert len(args.right_weights) == len(args.right_keys)
    assert len(args.img_or_txt[0]) == len(args.left_keys)
    assert len(args.img_or_txt[1]) == len(args.right_keys)
    if args.context_length:
        args.context_length = eval(args.context_length)
        assert len(args.img_or_txt[0]) == len(args.context_length[0])
        assert len(args.img_or_txt[1]) == len(args.context_length[1])
    return


def calculate_individual_err(qrels, retrieved_results):
    # Initialize ERR for the query
    ERR = 0.0
    # Initialize probability of user not stopping
    p_stop = 1.0

    # Sort documents by their retrieval scores in descending order
    sorted_docs = sorted(retrieved_results.items(), key=lambda x: x[1], reverse=True)
    max_rel = max([v for v in qrels.values()])
    # Iterate through sorted documents to calculate ERR
    for rank, (doc, _) in enumerate(sorted_docs, start=1):
        # Get the relevance grade for the document
        rel = qrels.get(doc, 0)
        # Normalize relevance score
        rel_prob = (2 ** rel - 1) / (2 ** max_rel)
        # Update ERR
        ERR += p_stop * rel_prob / rank
        # Update the probability of stopping
        p_stop *= (1 - rel_prob)
    return ERR


def calculate_mean_err(all_qrels, all_retrieved_results):
    total_ERR = 0.0
    num_queries = len(all_qrels)

    # Calculate ERR for each query and sum them
    for query, qrels in tqdm(all_qrels.items()):
        retrieved_results = all_retrieved_results.get(query, {})
        total_ERR += calculate_individual_err(qrels, retrieved_results)

    # Compute the mean ERR across all queries
    mean_ERR = total_ERR / num_queries if num_queries > 0 else 0
    return mean_ERR


def calculate_individual_rbp(qrels, retrieved_results, p):
    # Initialize RBP for the query
    RBP = 0.0

    # Sort documents by their retrieval scores in descending order
    sorted_docs = sorted(retrieved_results.items(), key=lambda x: x[1], reverse=True)
    max_rel = max([v for v in qrels.values()])
    # Iterate through sorted documents to calculate RBP
    for rank, (doc, _) in enumerate(sorted_docs, start=1):
        # Get the relevance grade for the document
        rel = qrels.get(doc, 0)
        # Normalize relevance score
        rel_norm = rel / max_rel
        # Update RBP
        RBP += rel_norm * (p ** (rank - 1))
    RBP *= (1-p)
    return RBP


def calculate_mean_rbp(qrels, retrieved_results, p=0.9):
    # Function to calculate individual RBP for a query

    # Calculate RBP for each query and average them
    total_RBP = 0.0
    num_queries = len(qrels)

    for query, docs_qrels in qrels.items():
        docs_retrieved_results = retrieved_results.get(query, {})
        total_RBP += calculate_individual_rbp(docs_qrels, docs_retrieved_results, p=p)

    # Compute the mean RBP across all queries
    mean_RBP = total_RBP / num_queries if num_queries > 0 else 0
    return mean_RBP


logging.basicConfig(level=logging.INFO)


def calc_all_features_mf(model_name, model, tokenizer, doc_meta_list, preprocess, args, side=0):
    all_features = []
    mmevaldataset = MFRightEvalDataset(doc_meta_list, tokenizer, preprocess, args, side=side)
    mmevaldataloader = DataLoader(mmevaldataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    for batch in tqdm(mmevaldataloader):

        rights, right_weights = batch
        right_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for j, cat in enumerate(args.img_or_txt[side]):
                if cat == "txt":
                    rights[j] = rights[j].cuda()
                    right_features.append(model.encode_text(rights[j], normalize=True))
                else:
                    rights[j] = rights[j].cuda()
                    right_features.append(model.encode_image(rights[j], normalize=True))

        right_features = torch.stack(right_features, dim=1)
        right_features_mean = (right_features * right_weights.unsqueeze(-1).repeat(1, 1, right_features.shape[-1]).to(device=right_features.device, dtype=right_features.dtype)).sum(1)
        right_features_mean = F.normalize(right_features_mean, dim=-1)
        if args.run_queries_cpu:
            all_features += right_features_mean.cpu()
        else:
            all_features += right_features_mean
    return all_features


def load_model(model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to('cuda')
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def get_test_queries(df_test, top_q=2000, weight_key=None, query_key="query"):
    _df_temp_ = df_test[[query_key, weight_key]]
    _df_temp_ = _df_temp_.groupby(query_key).sum()
    if top_q == -1:
        top_q = len(df_test[query_key].unique())
    sampled_data = _df_temp_.sample(n=top_q, weights=_df_temp_[weight_key], random_state=1, replace=False)
    # sampled_data = _df_temp_.sample(n=top_q, random_state=1)
    sampled_data = sampled_data.sort_values(by=weight_key, ascending=False)
    test_queries = list(sampled_data.index)
    return test_queries


def run_queries(test_queries, query_features, doc_ids_all, all_features, k):
    # now do the search
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    query_features = query_features.cuda()
    all_features = all_features.cuda()

    results = dict()
    for query, query_feature in tqdm(zip(test_queries, query_features)):

        similarity = cos(query_feature.unsqueeze(0).to(all_features.device), all_features)

        top_scores, top_inds = torch.topk(similarity, k)
        top_scores = list(top_scores.cpu().numpy())
        top_inds = list(top_inds.cpu().numpy())
        results[query] = {str(doc_ids_all[idx]): float(_s) for idx, _s in zip(top_inds, top_scores)}
    return results


def run_queries_cpu(test_queries, query_features, doc_ids_all, all_features, k):

    results = dict()
    all_features_np = all_features.cpu().numpy()
    index = faiss.IndexFlatIP(all_features_np.shape[1])
    index.add(all_features_np)

    query_features_np = query_features.cpu().numpy()
    top_scores_all, top_inds_all = index.search(query_features_np, k=k)
    for query, top_scores, top_inds in zip(test_queries, top_scores_all, top_inds_all):

        results[query] = {str(doc_ids_all[idx]): float(_s) for idx, _s in zip(top_inds, top_scores)}
    return results


def run_eval(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--doc-meta", type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--hf-dataset-split", type=str, default='data')
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--model_name", type=str, help="Model type", default="ViT-B-32")
    parser.add_argument("--preprocess", type=str, default=None)
    parser.add_argument("--overwrite-feature", action="store_true", default=False)
    parser.add_argument("--overwrite-retrieval", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight_key", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str)

    parser.add_argument("--img-or-txt", type=str, default="[['txt'], ['img', 'txt']]")
    parser.add_argument("--left-keys", type=str, default="['text']")
    parser.add_argument("--right-keys", type=str, default="['image_local', 'title']")
    parser.add_argument("--left-weights", type=str, default="[1]")
    parser.add_argument("--right-weights", type=str, default="[0.9, 0.1]")

    parser.add_argument("--query_ids_dict_path", type=str, default=None)
    parser.add_argument("--gt_results_path", type=str, default=None)

    parser.add_argument("--context-length", type=str, default=None, help="context-length")
    parser.add_argument("--top-q", type=int, default=2000)
    parser.add_argument("--doc-id-key", type=str, default="product_id")
    parser.add_argument("--query-id-key", type=str, default=None)
    parser.add_argument("--metric-only", action="store_true", default=False)

    parser.add_argument("--run-queries-cpu", action="store_true", default=False)
    parser.add_argument("--features-path", type=str, default=None)



    args = parser.parse_args(argv)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.features_path:
        args.features_path = os.path.join(args.output_dir, "features.pt")
    args.output_json = os.path.join(args.output_dir, "output.json")
    args.retrieval_path = os.path.join(args.output_dir, "retrieval.json")

    if not args.gt_results_path:
        args.gt_results_path = os.path.join(args.output_dir, "gt_results.json")


    additional_model_cfgs = scan_model_configs()
    open_clip.factory._MODEL_CONFIGS.update(additional_model_cfgs)


    process_multi_modal_args(args)

    # 1) If the there is context length from params, we should use that.
    # 2) If it is default as 0, we should use the length in model cfg.
    # 3) If there is no cfg and no param, use 77.
    if args.context_length:
        max_context_length = max(max(args.context_length[0]), max(args.context_length[1]))
    else:
        max_context_length = 0
    if max_context_length == 0:
        if 'context_length' in open_clip.factory._MODEL_CONFIGS[args.model_name]['text_cfg']:
            max_context_length = open_clip.factory._MODEL_CONFIGS[args.model_name]['text_cfg']['context_length']
        else:
            max_context_length = 77
    else:
        open_clip.factory._MODEL_CONFIGS[args.model_name]['text_cfg']['context_length'] = max_context_length
    args.context_length = max_context_length
    gt_results = None
    if not args.metric_only:
        model_name = args.model_name
        pretrained = args.pretrained
        logging.info(f"{model_name} {pretrained}")

        if pretrained and ".zip" in pretrained:
            model, preprocess, tokenizer = load_model(model_name, "")

            model_local_repo = pretrained.replace(".zip", "")
            with zipfile.ZipFile(pretrained, 'r') as zip_ref:
                zip_ref.extractall(model_local_repo)
            model.text.transformer = AutoModel.from_pretrained(model_local_repo)
            model = model.to('cuda')
            model.eval()
        else:
            model, preprocess, tokenizer = load_model(model_name, pretrained)

        if args.preprocess:
            _, preprocess, _ = load_model(model_name, args.preprocess)

        model = model.to('cuda')

        logging.info("loading df test")
        if args.hf_dataset is not None:
            hf_dataset = load_dataset(args.hf_dataset, num_proc=os.cpu_count())[args.hf_dataset_split]
            df_test = hf_dataset.to_pandas()
        else:
            df_test = pd.read_csv(args.test_csv)
        query_key = args.query_id_key
        if not query_key:
            query_key = "query_id"
            df_test[query_key] = ""
            for col in args.left_keys:
                df_test[query_key] += df_test[col] + "_{!@#~}_"

        logging.info(df_test)
        if (args.weight_key in df_test.columns) and len(df_test[args.weight_key].unique()) > 1:
            df_test[args.weight_key] = (((df_test[args.weight_key] - df_test[args.weight_key].min()) / (df_test[args.weight_key].max() - df_test[args.weight_key].min())) * 99 + 1).astype(int)
        else:
            args.weight_key = "score"
            df_test[args.weight_key] = 1

        # get the test queries and gt_results if it is there.
        if os.path.exists(args.gt_results_path):
            logging.info("Loading Ground Truth")
            with open(args.gt_results_path, "r") as f:
                gt_results = json.load(f)
                test_queries = list(gt_results.keys())
        else:
            test_queries = get_test_queries(df_test, top_q=args.top_q, weight_key=args.weight_key, query_key=query_key)

        df_test = df_test.set_index(query_key)
        df_test[args.doc_id_key] = df_test[args.doc_id_key].astype(str)
        # df_test['key'] = df_test.index + df_test[args.doc_id_key]

        # Get Query Meta and features.
        df_query = df_test[~df_test.index.duplicated()]
        query_meta_list = [df_query.loc[query_id].to_dict() for query_id in test_queries]
        query_features = calc_all_features_mf(model_name, model, tokenizer, query_meta_list, preprocess, args, side=0)
        query_features = torch.stack(query_features)

        # Get Doc_IDs and doc meta
        if args.hf_dataset is not None:
            doc_meta = hf_dataset.to_pandas().set_index(args.doc_id_key).to_dict('index')
        else:
            with open(args.doc_meta, "r") as f:
                doc_meta = json.load(f)
        doc_ids_all = []
        doc_meta_list = []
        for key in doc_meta:
            doc_ids_all.append(key)
            doc_meta_list.append(doc_meta[key])

        if not os.path.isfile(args.features_path) or args.overwrite_feature:
            all_features = calc_all_features_mf(model_name, model, tokenizer, doc_meta_list, preprocess, args, side=1)
            torch.save(all_features, args.features_path)
        else:
            all_features = torch.load(args.features_path)

        if all_features is not None:
            all_features = torch.stack(all_features)
            logging.info(f"{all_features.shape} {all_features.dtype}")


    # Get Ground truth Results
    if os.path.exists(args.gt_results_path) and gt_results is None:
        logging.info("Loading Ground Truth")
        with open(args.gt_results_path, "r") as f:
            gt_results = json.load(f)
    else:
        logging.info("Computing Ground Truth")
        gt_results = {}
        for query in tqdm(test_queries):
            _df_query = df_test.loc[[query]].sort_values(by=args.weight_key, ascending=False)
            relevant_docs, relevance = _df_query[args.doc_id_key][:1000].tolist(), _df_query[args.weight_key][:1000].tolist()
            gt_results[query] = {doc: round(rel) for doc, rel in zip(relevant_docs, relevance)}
        with open(args.gt_results_path, "w") as f:
            json.dump(gt_results, f)

    if os.path.exists(args.retrieval_path) and not args.overwrite_retrieval:
        logging.info("Loading Retrieval")
        with open(args.retrieval_path, "r") as f:
            retrieval_results = json.load(f)
    else:
        logging.info("Running Retrieval")
        if args.run_queries_cpu:
            retrieval_results = run_queries_cpu(test_queries, query_features, doc_ids_all, all_features, 1000)
        else:
            retrieval_results = run_queries(test_queries, query_features, doc_ids_all, all_features, 1000)
        with open(args.retrieval_path, "w") as f:
            json.dump(retrieval_results, f)


    # Evaluation Starts
    logging.info("Evaluation Starts")
    evaluator = EvaluateRetrieval()
    ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
    ndcg, _map, recall, precision = evaluator.evaluate(gt_results, retrieval_results, ks)
    output_results = {
        'mAP': _map,
        'ndcg': ndcg,
        'precision': precision,
        'recall': recall
    }
    more_metrics = ["mrr", "hole", "acc"]
    for metric in more_metrics:
        res = evaluator.evaluate_custom(gt_results, retrieval_results, ks, metric=metric)
        output_results[metric] = res

    logging.info("Measure ERR")
    mean_err = calculate_mean_err(gt_results, retrieval_results)
    mean_rbp_7 = calculate_mean_rbp(gt_results, retrieval_results, p=0.7)
    mean_rbp_8 = calculate_mean_rbp(gt_results, retrieval_results, p=0.8)
    mean_rbp_9 = calculate_mean_rbp(gt_results, retrieval_results, p=0.9)

    output_results["summary"] = {
        'mAP@1000': [_map["MAP@1000"]],
        'mrr@1000': [output_results['mrr']["MRR@1000"]],
        'NDCG@10': [ndcg["NDCG@10"]],
        'Recall@10': [recall["Recall@10"]],
        'P@10': [precision["P@10"]],
        'mERR': mean_err,
        'mRBP7': mean_rbp_7,
        'mRBP8': mean_rbp_8,
        'mRBP9': mean_rbp_9,
    }

    logging.info(output_results["summary"])
    output_sum_df = pd.DataFrame(output_results["summary"])
    logging.info(output_sum_df)
    output_sum_df.to_csv(args.output_json.replace(".json", ".csv"), index=False)

    with open(args.output_json, 'w') as f:
        json.dump(output_results, f)

    return output_results


if __name__ == "__main__":
    try:
        run_eval(sys.argv[1:])
    except Exception as e:
        logging.error(f"Unexpected error caught: {e}", exc_info=True)
