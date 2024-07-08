from eval_gs_v1 import run_eval
from itertools import islice
import os
import pytest
import requests
import tarfile
import tempfile

s3_bucket_url = "https://marqo-gcl-public.s3.amazonaws.com/gcl-test-data/tiny-dataset/v1"

@pytest.fixture(scope="session")
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"temp directory: {temp_dir}")
        yield temp_dir
        print(f"temp directory deleted: {temp_dir}")


# Download a file from a URL and save it to temp_dir
def download(temp_dir, url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we handle potential download errors
    file_name = os.path.join(temp_dir, os.path.basename(url))
    with open(file_name, "wb") as f:
        f.write(response.content)
    return file_name


# Extract a tar file into temp_dir
def extract_tar(temp_dir, file_path):
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=temp_dir)


# Print the first num_lines of a file
def print_first_lines(file_path, num_lines):
    with open(file_path, "r") as file:
        for line in islice(file, num_lines):
            print(line, end="")


# Replace a target string with a replacement string in a text file (assumes the file is small enough to fit in memory)
def replace_string_in_file(file_path, target_string, replacement_string):
    # Read the contents of the file
    with open(file_path, "r") as file:
        file_contents = file.read()

    modified_contents = file_contents.replace(target_string, replacement_string)

    # Write the modified contents back to the file
    with open(file_path, "w") as file:
        file.write(modified_contents)


@pytest.fixture(scope="session")
def test_csv_path(temp_dir):
    test_csv_path = download(temp_dir, f"{s3_bucket_url}/query_tiny.csv")
    # update image paths as GCL expects full paths
    replace_string_in_file(test_csv_path, '/input/images/', f'{temp_dir}/images/')
    return test_csv_path


@pytest.fixture(scope="session")
def doc_meta_path(temp_dir):
    doc_meta_path = download(temp_dir, f"{s3_bucket_url}/corpus_tiny.json")
    # update image paths as GCL expects full paths
    replace_string_in_file(doc_meta_path, '/input/images/', f'{temp_dir}/images/')
    return doc_meta_path


@pytest.fixture(scope="session")
def pretrained_path(temp_dir):
    return download(temp_dir, f"{s3_bucket_url}/marqo-gcl-vitl14-124-gs-full_states.pt")


@pytest.fixture(scope="session")
def images_path(temp_dir):
    return extract_tar(temp_dir, download(temp_dir, f"{s3_bucket_url}/images.tar"))


def test_run_eval( temp_dir, pretrained_path: str, test_csv_path: str, doc_meta_path: str, images_path: str):

    # Print the first few lines of the files to verify they are formatted correctly
    print_first_lines(test_csv_path, 3)
    print_first_lines(doc_meta_path, 3)

    result = run_eval(["--model_name", "ViT-L-14",
            "--test_csv", test_csv_path,
            "--doc-meta", doc_meta_path,
            "--weight_key", "score_linear",
            "--output-dir", f'{temp_dir}/output',
            "--pretrained", pretrained_path,
            "--batch-size", "512",
            "--num_workers", "2",
            "--left-keys", "['query']",
            "--right-keys", "['image_local']",
            "--img-or-txt", "[['txt'], ['img']]",
            "--left-weights", "[1]",
            "--right-weights", "[1]",
            "--context-length", "[[77], [0]]",
            "--top-q", "15"])

    # due to stochastic nature of the evaluation, we can't assert the exact values in the result so we'll just do
    # some basic checks:
    assert result, "The result should not be empty"
    assert isinstance(result, dict), "The result should be a dictionary"
    assert 'summary' in result, "The dictionary should have a 'summary' key"
    assert result['summary'], "The 'summary' key should not have an empty result"
