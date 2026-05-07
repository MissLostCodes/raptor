import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import re
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters

import re

def extract_section_signal(text):
    """
    Robust multi-domain section + semantic signal.
    Works across:
    - research papers (QASPER)
    - narratives (NarrativeQA)
    - long comprehension docs (QuALITY)

    Combines:
    - keyword categories
    - structural regex patterns
    - normalized scoring
    """

    text = text.lower()

    # ---------------- CATEGORY DEFINITIONS ----------------
    categories = {
        "academic_structure": [
            "introduction", "background", "related work", "method",
            "approach", "model", "experiment", "results",
            "discussion", "conclusion", "evaluation"
        ],
        "narrative_structure": [
            "chapter", "story", "scene", "character",
            "dialogue", "event", "plot", "conflict",
            "resolution", "ending"
        ],
        "qa_reasoning": [
            "because", "therefore", "however", "thus",
            "in conclusion", "as a result", "for example"
        ],
        "technical_content": [
            "algorithm", "system", "framework",
            "implementation", "performance", "dataset",
            "training", "testing", "accuracy"
        ],
        "analytical_content": [
            "analysis", "comparison", "evaluation",
            "metrics", "evidence", "findings",
            "hypothesis", "observation"
        ]
    }

    # ---------------- KEYWORD SCORING ----------------
    category_scores = []

    for keywords in categories.values():
        count = sum(text.count(k) for k in keywords)
        normalized = count / (len(keywords) + 1)
        category_scores.append(normalized)

    # ---------------- REGEX STRUCTURE SIGNAL ----------------
    pattern_features = [
        r"\d+\.",              # numbered sections (1. 2. 3.)
        r"chapter\s+\d+",      # chapter 1, chapter 2
        r"section\s+\d+",      # section 1
        r"\([a-z]\)",          # (a), (b)
        r"\bfigure\s+\d+",     # figure 1
        r"\btable\s+\d+"       # table 1
    ]

    pattern_score = sum(
        len(re.findall(p, text)) for p in pattern_features
    ) / 10.0

    # ---------------- FINAL AGGREGATION ----------------
    max_signal = max(category_scores) if category_scores else 0
    avg_signal = sum(category_scores) / len(category_scores) if category_scores else 0

    # combine keyword + structure signal
    final_score = 0.6 * max_signal + 0.25 * avg_signal + 0.15 * pattern_score

    return final_score

def extract_entity_density(text):
    """
    Proxy for reasoning relevance (PageIndex-like signal).
    """
    words = text.split()
    if len(words) == 0:
        return 0

    capital_words = sum(1 for w in words if w.istitle())
    return capital_words / len(words)


def build_hybrid_embedding(
    node,
    embedding_model_name,
    total_nodes,
    use_structure=True,
    use_reasoning=True,
    use_pageindex_signal=True,
    weights=None
):
    """
    FINAL HYBRID EMBEDDING
    Combines:
    - semantic embedding (RAPTOR)
    - structure signal (position, length)
    - PageIndex-style section signal
    - reasoning proxy
    """

    if weights is None:
        weights = {
            "structure": 0.15,
            "reasoning": 0.1,
            "pageindex": 0.15
        }

    # ---------------- BASE EMBEDDING ----------------
    text_emb = np.array(node.embeddings[embedding_model_name])

    features = []

    # ---------------- STRUCTURE SIGNAL ----------------
    if use_structure:
        position = node.index / total_nodes
        length = len(node.text) / 1000
        features.extend([position, length])

    # ---------------- PAGEINDEX STRUCTURE SIGNAL ----------------
    if use_pageindex_signal:
        section_score = extract_section_signal(node.text)
        features.append(section_score)

    # ---------------- REASONING SIGNAL ----------------
    if use_reasoning:
        entity_density = extract_entity_density(node.text)
        features.append(entity_density)

    # ---------------- FINAL COMBINATION ----------------
    if len(features) > 0:
        features = np.array(features)

        # scale features so they don’t dominate embeddings
        #features = features * 0.1
        scale = np.linalg.norm(text_emb) / (len(text_emb) + 1)
        features = features * scale

        hybrid = np.concatenate([text_emb, features])
    else:
        hybrid = text_emb

    return hybrid

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
            **kwargs
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        #embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        # ---------------- CONFIG ----------------
        use_structure = kwargs.get("use_structure", True)
        use_reasoning = kwargs.get("use_reasoning", False)
        use_pageindex_signal = kwargs.get("use_pageindex_signal", True)

        total_nodes = len(nodes)

        # ---------------- BUILD HYBRID EMBEDDINGS ----------------
        embeddings = np.array([
            build_hybrid_embedding(
                node,
                embedding_model_name,
                total_nodes,
                use_structure=use_structure,
                use_reasoning=use_reasoning,
                use_pageindex_signal=use_pageindex_signal
            )
            for node in nodes
        ])
        # Perform the clustering
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        max_length_in_cluster,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        verbose=verbose,
                        use_structure=use_structure,
                        use_reasoning=use_reasoning,
                        use_pageindex_signal=use_pageindex_signal
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
