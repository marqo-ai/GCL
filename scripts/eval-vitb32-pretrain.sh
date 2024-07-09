data_root="/path"
model_root="/path"

rm -rf $model_root/VITB32-Pretrain

mkdir $model_root/VITB32-Pretrain

python evals/eval_gs_v1.py \
      --model_name ViT-B-32 \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_0_product_id_0.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_1.json \
      --weight_key "score_linear" \
      --output-dir $model_root/VITB32-Pretrain/eval_train_e20 \
      --pretrained laion2b_s34b_b79k\
      --batch-size 1024 \
      --num_workers 4 \
      --left-keys "['query']" \
      --right-keys "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weights "[1]" \
      --right-weights "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name ViT-B-32 \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_0_product_id_1.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_2.json \
      --weight_key "score_linear" \
      --output-dir $model_root/VITB32-Pretrain/eval_q0d1_e20 \
      --pretrained laion2b_s34b_b79k\
      --batch-size 1024 \
      --num_workers 4 \
      --left-keys "['query']" \
      --right-keys "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weights "[1]" \
      --right-weights "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name ViT-B-32 \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_1_product_id_0.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_1.json \
      --weight_key "score_linear" \
      --output-dir $model_root/VITB32-Pretrain/eval_q1d0_e20 \
      --pretrained laion2b_s34b_b79k\
      --batch-size 1024 \
      --num_workers 4 \
      --left-keys "['query']" \
      --right-keys "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weights "[1]" \
      --right-weights "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/eval_gs_v1.py \
      --model_name ViT-B-32 \
      --test_csv $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/query_1_product_id_1.csv \
      --doc-meta $data_root/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_2.json \
      --weight_key "score_linear" \
      --output-dir $model_root/VITB32-Pretrain/eval_q1d1_e20 \
      --pretrained laion2b_s34b_b79k\
      --batch-size 1024 \
      --num_workers 4 \
      --left-keys "['query']" \
      --right-keys "['image_local']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weights "[1]" \
      --right-weights "[1]" \
      --context-length "[[77], [0]]" \
      --top-q 2000


python evals/aggregate_results.py \
      --exp-path $model_root/VITB32-Pretrain \
      --evals "['eval_train_e20', 'eval_q0d1_e20', 'eval_q1d0_e20', 'eval_q1d1_e20']"
