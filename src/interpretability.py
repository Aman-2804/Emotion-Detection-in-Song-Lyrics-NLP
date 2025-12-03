"""
Interpretability analysis for emotion detection models.
Includes attention visualization and embedding analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import List, Tuple, Optional
import numpy as np


def visualize_attention(
    attention_weights: List[torch.Tensor],
    tokens: List[str],
    layer_idx: int = -1,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights for a given layer and head.
    
    Args:
        attention_weights: List of attention tensors from all layers
        tokens: List of token strings
        layer_idx: Which layer to visualize (-1 for last layer)
        head_idx: Which attention head to visualize
        save_path: Path to save the visualization
    """
    if layer_idx < 0:
        layer_idx = len(attention_weights) + layer_idx
    
    # Extract attention for specified layer and head
    # Shape: [batch, heads, seq_len, seq_len]
    attention = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    
    # Average over query positions (focus on tokens that receive attention)
    avg_attention = attention.mean(axis=0)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        avg_attention,
        xticklabels=tokens[:len(avg_attention)],
        yticklabels=tokens[:len(avg_attention)],
        cmap='YlOrRd',
        cbar=True
    )
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: List[str],
    method: str = 'umap',
    save_path: Optional[str] = None,
    n_components: int = 2
):
    """
    Visualize emotion embeddings in 2D space using t-SNE or UMAP.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        labels: List of emotion labels for each embedding
        method: 'umap' or 'tsne'
        save_path: Path to save the visualization
        n_components: Number of dimensions for reduction (2 for visualization)
    """
    # Reduce dimensionality
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
    
    # Get unique labels and colors
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=label,
            c=[label_to_color[label]],
            alpha=0.6,
            s=50
        )
    
    plt.title(f'Emotion Embeddings Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    emotion_labels: List[str],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for emotion classification.
    
    Args:
        y_true: True emotion labels
        y_pred: Predicted emotion labels
        emotion_labels: Ordered list of all emotion categories
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )
    plt.title('Confusion Matrix - Emotion Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
