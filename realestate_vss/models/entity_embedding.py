from typing import List

import numpy as np

class CategoricalEmbedder:
  def __init__(self, categories: List[str], unk_token: str = 'UNK', name: str = None):
    """
    Initialize the CategoricalEmbedder with a list of categories and an optional unknown token.
    """
    self.categories = [unk_token] + categories  # Prepend unk_token such that 0 <-> 'UNK'
    self.index = {category: idx for idx, category in enumerate(self.categories)}   # Map category to index e.g. 'AB' -> 1
    self.embeddings = self._generate_embeddings()

    self.name = name

  def _generate_embeddings(self) -> np.ndarray:
    """
    Private method to generate the embedding matrix with one-hot encoding for each category,
    including an all-hot vector for the unknown token.
    """
    n_categories = len(self.categories)
    embeddings = np.zeros((n_categories, n_categories - 1))  # n-1 for actual categories excluding 'UNK'
    embeddings[0, :] = 1  # All-hot vector for 'UNK'
    for category, idx in self.index.items():
        if idx > 0:
            embeddings[idx, idx - 1] = 1
    return embeddings

  def tokenize(self, categories: List[str]) -> List[int]:
    """
    Convert a list of category names to their corresponding indices.
    """
    return [self.index.get(category, 0) for category in categories]  # Default to 'UNK' for unknown categories

  def get_embedding(self, category: str) -> np.ndarray:
    """
    Retrieve the embedding vector for a one given category.
    This method is esp. nice for embedding the querying category.
    """
    return self.embeddings[self.index.get(category, 0)].reshape(1, -1)  # 1 x n_categories

  def get_embeddings(self, categories: List[str]) -> np.ndarray:
    """
    Retrieve the embedding vectors for a list of categories.
    """
    ids = self.tokenize(categories)
    return self.embeddings[ids]
  
# Example usage:
# categories = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'ON', 'PE', 'QC', 'SK', 'YT']
# unknown_tokens = 'UNK'
# province_embedder = CategoricalEmbedder(categories, unk_token=unknown_tokens)

# print("Category Index:", province_embedder.index)
# print("Embedding Matrix:\n", province_embedder.embeddings)
# print("Tokenize Example:", province_embedder.tokenize(['AB', 'QC', 'Unknown', 'ON', 'XYZ']))