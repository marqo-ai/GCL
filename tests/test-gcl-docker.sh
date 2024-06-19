#!/bin/bash
# Note: this script is meant to be run on a Linux GPU machine with docker and nvidia-docker installed.
# It will download a small dataset and a pretrained model from S3, and run the GCL evaluation on it.
# It will print the evaluation results, which should be similar to the one in comments at the end of this script.


# Default Docker image name
DEFAULT_DOCKER_IMAGE_NAME="gcl"

# Function to display help message
show_help() {
    echo "Usage: $0 data [docker_image_name]"
    echo ""
    echo "Arguments:"
    echo "  data_root         Path to the data directory, where the input data will be downloaded and the output will be written."
    echo "                    Does not need to exist, it will be created if it does not."
    echo "  docker_image_name Optional. Name of the Docker image to build. Defaults to '$DEFAULT_DOCKER_IMAGE_NAME'."
}

# Function to ensure the Docker image exists or build it if it doesn't
ensure_docker_image() {
    local image_name="$1"
    local script_dir
    local dockerfile_dir

    # Check if the Docker image exists
    if docker image inspect "$image_name" >/dev/null 2>&1; then
        echo "The Docker image '$image_name' already exists."
    else
        echo "The Docker image '$image_name' does not exist. Building it..."
        # Get the directory of the Dockerfile which is the parent directory of the script directory:
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        dockerfile_dir="$(dirname "$script_dir")"

        # Build the Docker image
        docker build -t "$image_name" "$dockerfile_dir"

        if [ $? -eq 0 ]; then
            echo "The Docker image '$image_name' has been built successfully."
        else
            echo "Failed to build the Docker image '$image_name'."
            exit 1
        fi
    fi
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    show_help
    exit 1
fi

data_root=$1
docker_image_name=${2:-$DEFAULT_DOCKER_IMAGE_NAME}
ensure_docker_image "$docker_image_name"

s3_bucket_url="https://marqo-gcl-public.s3.amazonaws.com/gcl-test-data/tiny-dataset/v1"
data_set=$data_root/input && mkdir -p $data_set
model_path=$data_root/models && mkdir -p $model_path

csv_file=$data_set/query_tiny.csv
corpus_file=$data_set/corpus_tiny.json
images_tar=$data_set/images.tar
model_file=$model_path/marqo-gcl-vitl14-124-gs-full_states.pt

# if files do not exist, download them:
[[ ! -f $csv_file ]] && wget -O $csv_file $s3_bucket_url/query_tiny.csv
[[ ! -f $corpus_file ]] && wget $s3_bucket_url/corpus_tiny.json -O $corpus_file
[[ ! -f $model_file ]] && wget -O $model_file $s3_bucket_url/marqo-gcl-vitl14-124-gs-full_states.pt
[[ ! -f $images_tar ]] && wget -O $images_tar $s3_bucket_url/images.tar
[[ ! -d $data_set/images ]] && tar -xvf $images_tar -C $data_set >/dev/null 2>&1

eval_results=$data_root/evals
[[ -d $eval_results ]] && rm -rf $eval_results && mkdir -p $eval_results

container_name=gcl-test
docker rm -f $container_name >/dev/null 2>&1

docker run --rm -it --name $container_name --gpus all --ipc=host \
    -v $data_set:/input \
    -v $corpus_file:/input/corpus.json \
    -v $csv_file:/input/queries.csv \
    -v $model_file:/models/model.pt \
    -v $eval_results:/output \
    $docker_image_name \
    --model_name ViT-L-14 \
    --test_csv /input/queries.csv \
    --doc-meta /input/corpus.json \
    --weight_key "score_linear" \
    --output-dir /output \
    --pretrained /models/model.pt \
    --batch-size 512 \
    --num_workers 2 \
    --left-key "['query']" \
    --right-key "['image_local']" \
    --img-or-txt "[['txt'], ['img']]" \
    --left-weight "[1]" \
    --right-weight "[1]" \
    --context-length "[[77], [0]]" \
    --top-q 15

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
else
    echo "Evaluation results:"
    cat $eval_results/output.csv

    # we're expecting something like this:
    # mAP@1000,mrr@1000,NDCG@10,mERR,mRBP7,mRBP8,mRBP9
    # 0.28004,0.3813,0.2577,0.14147534799726885,0.13706266106295095,0.13631271010241755,0.11864885535755411
fi
