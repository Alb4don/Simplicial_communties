#identifying clique communities of simplicial complexes of arbitrary dimension
#python 3.6
#Sanjukta Krishnagopal s.krishnagopal@ucl.ac.uk
#August 2021

#!/usr/bin/env python3

import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import logging

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import null_space
import networkx as nx
import itertools
from functools import reduce
from collections import defaultdict, Counter
import random
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.offline
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available - visualization features disabled")

try:
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available - AMI calculation disabled")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplicialComplexAnalyzer:
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.graph = None
        self.simplices = None
        self.boundary_matrices = None
        self.removed_simplices = []
        
    def validate_graph(self, graph: nx.Graph) -> bool:
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input must be a NetworkX Graph")
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph cannot be empty")
        if not nx.is_connected(graph):
            logger.warning("Graph is not connected - results may be incomplete")
        return True
    
    def sanitize_input(self, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            return [self.sanitize_input(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, (int, float, np.integer, np.floating)):
            return float(data) if not np.isnan(data) and np.isfinite(data) else 0.0
        return str(data)
    
    def compute_incidence_matrices(self, graph: nx.Graph, remove_indices: Optional[List[int]] = None) -> Tuple[List[Dict], List[sparse.lil_matrix]]:
        self.validate_graph(graph)
        self.graph = graph
        
        try:
            cliques = list(nx.find_cliques(graph))
            cliques = [tuple(sorted(c)) for c in cliques]
            
            if not cliques:
                raise ValueError("No cliques found in graph")
            
            max_clique_size = max(len(c) for c in cliques)
            simplices = []
            
            for k in range(max_clique_size):
                k_simplices = set()
                for clique in cliques:
                    k_simplices.update(itertools.combinations(clique, k + 1))
                
                if not k_simplices:
                    break
                    
                k_simplices = sorted(k_simplices)
                simplices.append({simplex: idx for idx, simplex in enumerate(k_simplices)})
            
            boundary_matrices = [None] * len(simplices)
            boundary_matrices[0] = lil_matrix((1, graph.number_of_nodes()))
            
            for k in range(1, len(simplices)):
                num_k_minus_1 = len(simplices[k - 1])
                num_k = len(simplices[k])
                
                if num_k_minus_1 == 0 or num_k == 0:
                    boundary_matrices[k] = lil_matrix((num_k_minus_1, num_k))
                    continue
                
                boundary_matrices[k] = lil_matrix((num_k_minus_1, num_k))
                sign_vector = np.array([(-1) ** i for i in range(k + 1)])
                
                for k_simplex, j in simplices[k].items():
                    try:
                        boundary_indices = []
                        for s in itertools.combinations(k_simplex, k):
                            if s in simplices[k - 1]:
                                boundary_indices.append(simplices[k - 1][s])
                        
                        if len(boundary_indices) == len(sign_vector):
                            boundary_matrices[k][boundary_indices, j] = sign_vector
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Issue processing simplex {k_simplex}: {e}")
                        continue
            
            if remove_indices:
                self._remove_simplices(simplices, boundary_matrices, remove_indices)
            
            self._validate_boundary_property(boundary_matrices)
            self.simplices = simplices
            self.boundary_matrices = boundary_matrices
            
            betti_numbers = self._compute_betti_numbers(boundary_matrices)
            logger.info(f"Betti numbers: {betti_numbers}")
            
            return simplices, boundary_matrices
            
        except Exception as e:
            logger.error(f"Error computing incidence matrices: {e}")
            raise
    
    def _remove_simplices(self, simplices: List[Dict], boundary_matrices: List[sparse.lil_matrix], 
                         remove_indices: List[int]) -> None:
        if len(simplices) < 3:
            return
            
        remove_indices = sorted(set(remove_indices), reverse=True)
        simplex_flip = {v: k for k, v in simplices[2].items()}
        
        for idx in remove_indices:
            if 0 <= idx < len(simplices[2]):
                boundary_matrices[2][:, idx] = 0
                if idx in simplex_flip:
                    removed_simplex = simplex_flip[idx]
                    simplices[2].pop(removed_simplex)
                    self.removed_simplices.append(removed_simplex)
    
    def _validate_boundary_property(self, boundary_matrices: List[sparse.lil_matrix]) -> None:
        for k in range(1, len(boundary_matrices) - 1):
            if boundary_matrices[k] is not None and boundary_matrices[k + 1] is not None:
                try:
                    product = boundary_matrices[k].dot(boundary_matrices[k + 1])
                    if product.nnz > 0:
                        max_val = np.abs(product.data).max()
                        if max_val > self.tolerance:
                            logger.warning(f"Boundary property violation at level {k}: max value {max_val}")
                except Exception as e:
                    logger.warning(f"Could not validate boundary property at level {k}: {e}")
    
    def _compute_betti_numbers(self, boundary_matrices: List[sparse.lil_matrix]) -> List[int]:
        try:
            ranks = []
            for matrix in boundary_matrices:
                if matrix is not None and matrix.nnz > 0:
                    ranks.append(np.linalg.matrix_rank(matrix.todense()))
                else:
                    ranks.append(0)
            
            null_spaces = []
            for i, matrix in enumerate(boundary_matrices):
                if matrix is not None:
                    null_spaces.append(matrix.shape[1] - ranks[i])
                else:
                    null_spaces.append(0)
            
            betti_numbers = []
            for i in range(len(null_spaces) - 1):
                betti = null_spaces[i] - ranks[i + 1] if i + 1 < len(ranks) else null_spaces[i]
                betti_numbers.append(max(0, betti))
            
            return betti_numbers
        except Exception as e:
            logger.error(f"Error computing Betti numbers: {e}")
            return []
    
    def find_duplicate_indices(self, sequence: List[Any]) -> Dict[Any, List[int]]:
        tally = defaultdict(list)
        for i, item in enumerate(sequence):
            tally[item].append(i)
        return {key: locs for key, locs in tally.items() if len(locs) > 1}
    
    def check_support_overlap(self, support_vectors: List[List[int]]) -> List[Tuple[int, int]]:
        overlaps = []
        for i1 in range(len(support_vectors)):
            for i2 in range(i1):
                s1, s2 = set(support_vectors[i1]), set(support_vectors[i2])
                if s1 != s2 and s1.intersection(s2):
                    if len(s1) < len(s2):
                        overlaps.append((i1, i2))
                    elif len(s2) < len(s1):
                        overlaps.append((i2, i1))
        return overlaps
    
    def create_graph_from_adjacency(self, adjacency_matrix: np.ndarray) -> nx.Graph:
        if not isinstance(adjacency_matrix, np.ndarray):
            raise TypeError("Adjacency matrix must be numpy array")
        
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        adjacency_matrix = np.where(np.isfinite(adjacency_matrix), adjacency_matrix, 0)
        np.fill_diagonal(adjacency_matrix, 0)
        
        return nx.from_numpy_array(adjacency_matrix)
    
    def preprocess_graph(self, graph: nx.Graph, weight_percentile: float = 50.0, 
                        degree_multiplier: float = 3.0) -> nx.Graph:
        graph = graph.copy()
        
        if nx.is_weighted(graph):
            weights = [w for _, _, w in graph.edges(data="weight")]
            if weights:
                threshold = np.percentile(weights, weight_percentile)
                edges_to_remove = [(u, v) for u, v, w in graph.edges(data="weight") if w < threshold]
                graph.remove_edges_from(edges_to_remove)
        
        if graph.number_of_nodes() > 0:
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            nodes_to_remove = [node for node, degree in graph.degree() if degree < avg_degree * degree_multiplier]
            graph.remove_nodes_from(nodes_to_remove)
        
        graph.remove_nodes_from(list(nx.isolates(graph)))
        
        return graph
    
    def compute_hodge_laplacians(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]]]:
        if self.boundary_matrices is None:
            raise ValueError("Must compute incidence matrices first")
        
        clique_communities = []
        
        for i in range(len(self.boundary_matrices) - 1):
            try:
                if i + 1 < len(self.boundary_matrices) and self.boundary_matrices[i + 1] is not None:
                    laplacian_up = self.boundary_matrices[i + 1].dot(self.boundary_matrices[i + 1].T)
                else:
                    laplacian_up = sparse.lil_matrix((self.boundary_matrices[i].shape[0], 
                                                    self.boundary_matrices[i].shape[0]))
                
                if self.boundary_matrices[i] is not None:
                    laplacian_down = self.boundary_matrices[i].T.dot(self.boundary_matrices[i])
                else:
                    continue
                
                laplacian_hodge = laplacian_up + laplacian_down
                
                if laplacian_up.nnz > 0:
                    eigenvals, eigenvecs = self._compute_safe_eigenvalues(laplacian_up.todense())
                    communities = self._extract_communities(eigenvals, eigenvecs, i)
                    clique_communities.append(communities)
                else:
                    clique_communities.append([])
                    
            except Exception as e:
                logger.error(f"Error computing Hodge Laplacian at level {i}: {e}")
                clique_communities.append([])
        
        return clique_communities
    
    def _compute_safe_eigenvalues(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if matrix.shape[0] == 0:
                return np.array([]), np.array([]).reshape(0, 0)
            
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            valid_indices = np.isfinite(eigenvals) & np.isfinite(eigenvecs).all(axis=0)
            
            return eigenvals[valid_indices], eigenvecs[:, valid_indices]
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return np.array([]), np.array([]).reshape(matrix.shape[0], 0)
    
    def _extract_communities(self, eigenvals: np.ndarray, eigenvecs: np.ndarray, level: int) -> List[List[int]]:
        if len(eigenvals) == 0:
            return []
        
        nonzero_mask = eigenvals > self.tolerance
        if not np.any(nonzero_mask):
            return []
        
        nonzero_eigenvals = eigenvals[nonzero_mask]
        nonzero_eigenvecs = eigenvecs[:, nonzero_mask]
        
        supports = []
        for j in range(len(nonzero_eigenvals)):
            support = np.where(np.abs(nonzero_eigenvecs[:, j]) > self.tolerance)[0].tolist()
            if support:
                supports.append(support)
        
        if not supports:
            return []
        
        supports = self._handle_degenerate_eigenvalues(nonzero_eigenvals, supports)
        communities = self._extract_node_communities(supports, level)
        
        return [comm for comm in communities if len(comm) > level + 1]
    
    def _handle_degenerate_eigenvalues(self, eigenvals: np.ndarray, supports: List[List[int]]) -> List[List[int]]:
        rounded_eigenvals = np.round(eigenvals, 5)
        duplicates = self.find_duplicate_indices(rounded_eigenvals)
        
        for eigenval, indices in duplicates.items():
            if len(indices) > 1:
                union_support = set()
                for idx in indices:
                    if idx < len(supports):
                        union_support.update(supports[idx])
                
                union_list = list(union_support)
                for idx in indices:
                    if idx < len(supports):
                        supports[idx] = union_list
        
        unique_supports = []
        seen = set()
        for support in supports:
            support_tuple = tuple(sorted(support))
            if support_tuple not in seen:
                unique_supports.append(support)
                seen.add(support_tuple)
        
        return unique_supports
    
    def _extract_node_communities(self, supports: List[List[int]], level: int) -> List[List[int]]:
        if level >= len(self.simplices):
            return []
        
        simplex_to_nodes = {v: k for k, v in self.simplices[level].items()}
        communities = []
        
        for support in supports:
            if level == 0:
                community_nodes = support
            else:
                node_set = set()
                for simplex_idx in support:
                    if simplex_idx in simplex_to_nodes:
                        simplex = simplex_to_nodes[simplex_idx]
                        node_set.update(simplex)
                community_nodes = list(node_set)
            
            if community_nodes:
                communities.append(sorted(community_nodes))
        
        return communities
    
    def compute_adjusted_mutual_information(self, communities: List[List[int]], 
                                          true_labels: List[int], num_samples: int = 100) -> List[float]:
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - AMI calculation skipped")
            return []
        
        if not communities or not true_labels:
            return []
        
        node_to_communities = defaultdict(list)
        for comm_idx, community in enumerate(communities):
            for node in community:
                if node < len(true_labels):
                    node_to_communities[node].append(comm_idx)
        
        ami_scores = []
        for _ in range(min(num_samples, self.max_iterations)):
            predicted_labels = []
            for node in range(len(true_labels)):
                if node in node_to_communities and node_to_communities[node]:
                    predicted_labels.append(random.choice(node_to_communities[node]))
                else:
                    predicted_labels.append(-1)
            
            try:
                ami = adjusted_mutual_info_score(predicted_labels, true_labels)
                if np.isfinite(ami):
                    ami_scores.append(ami)
            except Exception as e:
                logger.warning(f"AMI computation failed: {e}")
                continue
        
        return ami_scores
    
    def visualize_graph(self, output_path: Optional[str] = None, 
                       layout_algorithm: str = 'spring') -> Optional[str]:
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - visualization skipped")
            return None
        
        if self.graph is None:
            raise ValueError("No graph available for visualization")
        
        try:
            if layout_algorithm == 'spring':
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif layout_algorithm == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)
            
            edge_trace = self._create_edge_trace(pos)
            node_trace = self._create_node_trace(pos)
            
            layout = go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Simplicial Complex Visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
            
            if output_path:
                safe_path = self._sanitize_path(output_path)
                fig.write_html(safe_path)
                logger.info(f"Visualization saved to {safe_path}")
                return safe_path
            else:
                return fig.to_html()
                
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
    
    def _sanitize_path(self, path: str) -> str:
        path = os.path.normpath(path)
        if not path.endswith(('.html', '.pdf', '.png', '.jpg', '.svg')):
            path += '.html'
        
        safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./\\')
        sanitized = ''.join(c if c in safe_chars else '_' for c in path)
        
        return sanitized
    
    def _create_edge_trace(self, pos: Dict[int, np.ndarray]) -> go.Scatter:
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
    def _create_node_trace(self, pos: Dict[int, np.ndarray]) -> go.Scatter:
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        
        node_adjacencies = [len(list(self.graph.neighbors(node))) for node in self.graph.nodes()]
        node_text = [f'Node {node}<br>Connections: {adj}' 
                    for node, adj in zip(self.graph.nodes(), node_adjacencies)]
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_adjacencies,
                size=10,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    title="Node Connections",
                    x=1.02
                ),
                line=dict(width=2)
            )
        )

def create_karate_club_example() -> Tuple[nx.Graph, List[int]]:
    graph = nx.karate_club_graph()
    true_labels = [graph.nodes[i]['club'] for i in graph.nodes()]
    return graph, true_labels

def create_custom_adjacency_example() -> Tuple[nx.Graph, List[int]]:
    adjacency = np.array([
        [0,0,0,0,1,1,0,0,0,0,0,0,1],
        [0,0,0,0,0,1,1,0,1,1,1,1,0],
        [0,0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,1,0,0,0,0,0],
        [1,0,0,1,0,1,0,1,0,0,0,0,0],
        [1,1,0,0,1,0,1,0,1,0,0,0,0],
        [0,1,1,0,0,1,0,1,0,0,0,0,0],
        [0,0,1,1,1,0,1,0,0,0,0,0,0],
        [0,1,0,0,0,1,0,0,0,1,1,0,0],
        [0,1,0,0,0,0,0,0,1,0,1,1,0],
        [0,1,0,0,0,0,0,0,1,1,0,1,0],
        [0,1,0,0,0,0,0,0,0,1,1,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
    
    analyzer = SimplicialComplexAnalyzer()
    graph = analyzer.create_graph_from_adjacency(adjacency)
    true_labels = list(range(len(adjacency)))
    
    return graph, true_labels

def main():
    try:
        analyzer = SimplicialComplexAnalyzer()
        
        graph, true_labels = create_karate_club_example()
        
        logger.info(f"Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        processed_graph = analyzer.preprocess_graph(graph)
        logger.info(f"Processed graph: {processed_graph.number_of_nodes()} nodes, {processed_graph.number_of_edges()} edges")
        
        simplices, boundary_matrices = analyzer.compute_incidence_matrices(processed_graph, remove_indices=[25])
        
        communities = analyzer.compute_hodge_laplacians()
        
        for level, level_communities in enumerate(communities):
            logger.info(f"Level {level} communities: {len(level_communities)}")
            for i, community in enumerate(level_communities):
                logger.info(f"  Community {i}: {community}")
        
        if communities and len(communities) > 1:
            ami_scores = analyzer.compute_adjusted_mutual_information(
                communities[1], 
                [true_labels[i] for i in processed_graph.nodes() if i < len(true_labels)]
            )
            if ami_scores:
                logger.info(f"AMI scores - Mean: {np.mean(ami_scores):.3f}, Std: {np.std(ami_scores):.3f}")
        
        output_file = analyzer.visualize_graph("simplicial_complex_output.html")
        if output_file:
            logger.info(f"Visualization completed: {output_file}")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
