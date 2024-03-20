# Generalized Contrastive Learning for Multi-Modal Retrieval and Ranking
This is a Repository for Marqo-GS-10M multi-modal fine-grained ranking dataset and benchmark followed by our novel training framework: Generalized Contrastive Learning (GCL).

## Motivation
Contrastive learning has gained widespread adoption for retrieval tasks due to its minimal requirement for manual annotations. However, popular contrastive frameworks typically learn from binary relevance, making them ineffective at incorporating direct fine-grained rankings. 

In this paper, we curate a large-scale dataset: Marqo-GS-10M, featuring detailed relevance scores for each query-document pair to facilitate future research and evaluation. 

Subsequently, we propose Generalized Contrastive Learning for Multi-Modal Retrieval and Ranking (GCL), which is designed to learn from fine-grained rankings beyond binary relevance score. 

Our results show that GCL achieves a **94.5%** increase in NDCG@10 for in-domain and **26.3** to **48.8%** increases for cold-start evaluations, measured **relative** to the CLIP baseline within our curated ranked dataset.

## Training Framework (GCL)
![Main Figure](docs/main_figure1.png)
Overview of our Generalized Contrastive Learning (GCL) approach. 
GCL integrates ranking information alongside multiple input fields for each sample (e.g., title and image) 
across both left-hand-side (LHS) and right-hand-side (RHS). 
Ground-truth ranking scores are transformed into weights, 
which are used for computing contrastive losses, ensuring that pairs with higher weights incur greater penalties.



## Dataset

### Multi-Split 

<img src="docs/ms1.png" alt="multi split visual" width="500"/>

### Dataset Visualization
![Dataset Qualitative](docs/visual_dataset_4.png)


## Downloads
### Dataset
The Marqo-GS-10M dataset is available for direct download. This dataset is pivotal for training and benchmarking in Generalized Contrastive Learning (GCL) frameworks and other multi-modal fine-grained ranking tasks.

- **Full Dataset**: [Download](https://marqo-gcl-public.s3.amazonaws.com/v1/marqo-gs-dataset.tar) - Link contains the entire Marqo-GS-10M dataset except for the images. 
- **Full Images**: [Download](https://marqo-gcl-public.s3.amazonaws.com/v1/images_archive.tar) - Link contains the entire Marqo-GS-10M dataset except for the images.
- **Sample Images**: [Download](https://marqo-gcl-public.s3.amazonaws.com/v1/images_wfash.tar) - Link contains the images for woman fashion category, it corresponds to the woman fashion sub-dataset. 

### Finetuned Models
- **VITL14**
- **VITB32**