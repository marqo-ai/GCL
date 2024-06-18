import os
import pytest
import requests
import subprocess
import sys
import tarfile
import tempfile


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
    test_csv_path = download(
        temp_dir,
        "https://marqtune-public-bucket.s3.amazonaws.com/test-data/gcl-test-data/tiny-dataset/query_tiny.csv",
    )
    # update image paths as GCL expects full paths
    replace_string_in_file(test_csv_path, '","images/', f'","{temp_dir}/images/')
    return test_csv_path


@pytest.fixture(scope="session")
def doc_meta_path(temp_dir):
    doc_meta_path = download(
        temp_dir,
        "https://marqtune-public-bucket.s3.amazonaws.com/test-data/gcl-test-data/tiny-dataset/corpus_tiny.json",
    )
    # update image paths as GCL expects full paths
    replace_string_in_file(
        doc_meta_path, '"image_local":"images/', f'"image_local": "{temp_dir}/images/'
    )
    return doc_meta_path


@pytest.fixture(scope="session")
def pretrained_path(temp_dir):
    return download(
        temp_dir,
        "https://marqtune-public-bucket.s3.amazonaws.com/test-data/gcl-test-data/tiny-dataset/marqo-gcl-vitl14-124-gs-full_states.pt",
    )


@pytest.fixture(scope="session")
def images_path(temp_dir):
    return extract_tar(
        temp_dir,
        download(
            temp_dir,
            "https://marqtune-public-bucket.s3.amazonaws.com/test-data/gcl-test-data/tiny-dataset/images.tar",
        ),
    )


def test_eval( temp_dir, pretrained_path: str, test_csv_path: str, doc_meta_path: str, images_path: str):
    process = subprocess.run([sys.executable,
                            "evals/eval_gs_v1.py",
                            "--model_name", "ViT-L-14",
                            "--test_csv", test_csv_path,
                            "--doc-meta", doc_meta_path,
                            "--weight_key", "score_linear",
                            "--output-dir", f'{temp_dir}/output',
                            "--pretrained", pretrained_path,
                            "--batch-size", "512",
                            "--num_workers", "2",
                            "--left-key", "['query']",
                            "--right-key", "['image_local']",
                            "--img-or-txt", "[['txt'], ['img']]",
                            "--left-weight", "[1]",
                            "--right-weight", "[1]",
                            "--context-length", "[[77], [0]]",
                            "--top-q", "15"], check=True)
    # Check return code
    assert process.returncode == 0
