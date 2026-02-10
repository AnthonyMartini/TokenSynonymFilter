# Token Synonym Filter

## Abstract
This project examines the necessity for fine-tuned classification models to retain embeddings for synonyms and logically similar tokens. The central premise is that discerning subtle differences between semantically equivalent words (e.g., "film" vs. "movie") is not essential for successful sentiment classification. By training a model *de novo* with a vocabulary where similar tokens are consolidated, we aim to reduce vocabulary size and computational overhead while maintaining high performance.

## Project Structure

*   **`ModelTraining.ipynb`**: Handles the loading of the tokenixer and training of the models on Wikitext and fine-tuning on the IMDb dataset.
*   **`NLTK_Version1.ipynb`**: Implements the **Version 1 (Strict Equivalence)** strategy using NLTK WordNet to identify and remove interchangeable tokens.
*   **`NLTK_Version2.ipynb`**: Implements the **Version 2 (Subset-Inclusion)** strategy, which uses a hierarchical reduction approach to remove redundant tokens.
*   **`TokenRefinement.ipynb`**: A notebook refactoring of the vocabulary refinement logic, identifying and removing highly interchangeable tokens based on contextual similarity and edit distance.
*   **`TokenFilter.ipynb`**: Contains logic for filtering the tokenizer's vocabulary based on embedding similarity and specific filter criteria. Not being used in the final model as results from this notebook were not promising.

## Methodology

### 1. Base Model
*   **Tokenizer**: WordPiece (30,000 tokens).
*   **Training**: Pretrained on Wikitext-103, fine-tuned on IMDb Large Movie Review Dataset.
*   **Architecture**: 8 Hidden Layers, 8 Attention Heads, 512 Hidden Size.

### 2. NLTK-Consolidated Model (Version 1: Strict Equivalence)
*   **Concept**: identifying pairs with a Jaccard Similarity Coefficient of 1.0 using NLTK WordNet synsets (strict semantic equivalence).
*   **Result**: Removed **2,186** redundant tokens (~7.3% reduction).
*   **Limitation**: Failed to capture synonyms where one term is polysemous (e.g., "movie" vs. "film").

### 3. Subset-Inclusion Model (Version 2: Hierarchical Reduction)
*   **Concept**: Shifts from strict equivalence to **subset inclusion**. Recognizes when one word's meaning is fully contained within another's (e.g., "movie" is a subset of "film").
*   **Algorithm**: Uses Iterative Mapping, Chain Resolution (Back-Propagation), and Forward-Propagation.
*   **Result**: Removed **3,910** tokens (~13% reduction).

## Datasets
*   **Pretraining**: [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) (100M+ tokens).
*   **Fine-tuning**: [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25k training, 25k testing).

## Performance Results (Summary)

All models were trained start to finish 3 times from scratch and evaluated on the IMDb test set (25,000 examples). These results are the averages of the 3 runs.

| Metric | Base Model (Avg) | Version 1 (Avg) | Version 2 (Avg) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8739 | 0.8753 | 0.8757 |
| **Precision** | 0.8680 | 0.8717 | 0.8715 |
| **F1 Score** | 0.8749 | 0.8759 | 0.8763 |
| **Recall** | 0.8820 | 0.8802 | 0.8813 |

*Note: The Version 2 model achieved a significant vocabulary reduction (13%) while slightly improving accuracy compared to the Base Model, supporting the hypothesis that semantic redundancy can be reduced without performance loss.*

## Environment
*   **Hardware**: NVIDIA H100 GPU (Google Colab).
*   **Attention Implementation**: Flash Attention 2.
