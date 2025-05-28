import unittest
import numpy as np
import networkx as nx
from src.missing_links_analyzer.similarity_calculator import (
    calculate_jaccard_similarity,
    get_common_neighbors,
    get_total_neighbors,
    get_jaccard_coefficient
)

class TestSimilarityCalculator(unittest.TestCase):
    def setUp(self):
        # Example simple graph for multiple tests
        self.simple_graph = nx.Graph()
        # Nodes: 0, 1, 2, 3. Sorted order will be [0, 1, 2, 3] by calculate_jaccard_similarity
        self.simple_graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
        
        # Graph with no edges
        self.no_edges_graph = nx.Graph()
        self.no_edges_graph.add_nodes_from([0, 1, 2]) # Sorted order [0, 1, 2]

        # Empty graph
        self.empty_graph = nx.Graph()

        # Expected matrices for simple_graph based on REFACTORED logic of similarity_calculator.py
        # Adjacency for [0,1,2,3]:
        self.adj_expected_simple = np.array([
            [0,1,1,0],
            [1,0,1,0],
            [1,1,0,1],
            [0,0,1,0]], dtype=int)

        # CN = (adj @ adj) with diag set to 0.
        # A@A = [[2,1,1,1], [1,2,1,1], [1,1,3,0], [1,1,0,1]]
        self.cn_expected_simple_refactored = np.array([
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,0], 
            [1,1,0,0]], dtype=int) 

        # TN_refactored = |N(i)|+|N(j)|-CN(i,j), with diag set to 1 by get_total_neighbors
        # Degrees: d0=2, d1=2, d2=3, d3=1
        # TN(0,1) = 2+2-1=3; TN(0,2)=2+3-1=4; TN(0,3)=2+1-1=2
        # TN(1,2)=2+3-1=4; TN(1,3)=2+1-1=2
        # TN(2,3)=3+1-0=4 (CN(2,3) is 0)
        self.tn_expected_simple_refactored = np.array([
            [1,3,4,2],
            [3,1,4,2],
            [4,4,1,4],
            [2,2,4,1]], dtype=int)

        # Jaccard = CN/TN_refactored (element-wise). Facade then sets diag to 1.0.
        # Raw J(0,1)=1/3; J(0,2)=1/4; J(0,3)=1/2
        # J(1,2)=1/4; J(1,3)=1/2
        # J(2,3)=0/4=0
        # Diags are 0/1=0 from get_jaccard_coefficient
        # Final Jaccard after facade sets diag to 1.0:
        self.jac_expected_simple_refactored_facade = np.array([
            [1.0, 1/3, 1/4, 1/2],
            [1/3, 1.0, 1/4, 1/2],
            [1/4, 1/4, 1.0, 0.0],
            [1/2, 1/2, 0.0, 1.0]], dtype=float)


    def test_get_common_neighbors_manual_refactored_logic(self):
        """Test get_common_neighbors with manually defined adj matrix based on refactored logic."""
        adj = self.adj_expected_simple
        expected_cn = self.cn_expected_simple_refactored
        common_neighbors = get_common_neighbors(adj)
        np.testing.assert_array_equal(common_neighbors, expected_cn)

    def test_get_total_neighbors_manual_refactored_logic(self):
        """Test get_total_neighbors with manually defined adj and CN based on refactored logic."""
        adj = self.adj_expected_simple
        cn_matrix = self.cn_expected_simple_refactored
        expected_tn = self.tn_expected_simple_refactored
        total_neighbors = get_total_neighbors(adj, cn_matrix)
        np.testing.assert_array_equal(total_neighbors, expected_tn)

    def test_get_jaccard_coefficient_manual_refactored_logic(self):
        """Test get_jaccard_coefficient with manual CN and TN based on refactored logic."""
        cn_matrix = self.cn_expected_simple_refactored
        tn_matrix = self.tn_expected_simple_refactored
        
        # Expected Jaccard = CN/TN before facade sets diag to 1.0
        # Diagonals are CN_ii/TN_ii = 0/1 = 0
        expected_jac_raw = np.array([
            [0.0, 1/3, 1/4, 1/2],
            [1/3, 0.0, 1/4, 1/2],
            [1/4, 1/4, 0.0, 0.0],
            [1/2, 1/2, 0.0, 0.0]], dtype=float)
            
        jaccard_matrix = get_jaccard_coefficient(cn_matrix, tn_matrix)
        np.testing.assert_array_almost_equal(jaccard_matrix, expected_jac_raw, decimal=7)

    def test_calculate_jaccard_empty_graph(self):
        """Test calculate_jaccard_similarity with an empty graph."""
        adj, cn, tn, jac = calculate_jaccard_similarity(self.empty_graph)
        self.assertEqual(adj.shape, (0,0))
        self.assertEqual(cn.shape, (0,0))
        self.assertEqual(tn.shape, (0,0)) 
        self.assertEqual(jac.shape, (0,0))

    def test_calculate_jaccard_graph_no_edges(self):
        """Test calculate_jaccard_similarity with a graph that has nodes but no edges."""
        # Nodes 0, 1, 2. Sorted order [0, 1, 2]
        adj, cn, total_neighbors_matrix, jac = calculate_jaccard_similarity(self.no_edges_graph)
        
        expected_adj = np.zeros((3,3), dtype=int)
        expected_cn = np.zeros((3,3), dtype=int) 
        
        # TN_refactored: degs are all 0. CN is all 0. So |N(i)|+|N(j)|-CN(i,j) = 0. Diag set to 1.
        expected_tn_refactored_no_edges = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=int)
        
        # Jaccard: Raw Jaccard is 0/0 (NaN) for off-diag, 0/1 for diag.
        # Facade `nan_to_num` converts NaN to 0. Then sets diag to 1.0.
        expected_jac_no_edges = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]], dtype=float)
        
        np.testing.assert_array_equal(adj, expected_adj)
        np.testing.assert_array_equal(cn, expected_cn)
        np.testing.assert_array_equal(total_neighbors_matrix, expected_tn_refactored_no_edges)
        np.testing.assert_array_almost_equal(jac, expected_jac_no_edges, decimal=7)

    def test_simple_graph_output_shapes(self):
        """Test output shapes for the simple graph."""
        adj, cn, tn, jac = calculate_jaccard_similarity(self.simple_graph)
        num_nodes = self.simple_graph.number_of_nodes()
        self.assertEqual(adj.shape, (num_nodes, num_nodes))
        self.assertEqual(cn.shape, (num_nodes, num_nodes))
        self.assertEqual(tn.shape, (num_nodes, num_nodes))
        self.assertEqual(jac.shape, (num_nodes, num_nodes))

    def test_simple_graph_jaccard_values(self): # Renamed from _original_logic
        """Test specific matrix values for the simple graph using calculate_jaccard_similarity (refactored logic)."""
        adj, cn, tn, jac = calculate_jaccard_similarity(self.simple_graph)
        
        np.testing.assert_array_equal(adj, self.adj_expected_simple)
        np.testing.assert_array_equal(cn, self.cn_expected_simple_refactored)
        np.testing.assert_array_equal(tn, self.tn_expected_simple_refactored)
        np.testing.assert_array_almost_equal(jac, self.jac_expected_simple_refactored_facade, decimal=7)

    def test_jaccard_diagonal_is_one(self):
        """Test that the diagonal of the Jaccard matrix from the facade is 1.0."""
        _, _, _, jac_simple = calculate_jaccard_similarity(self.simple_graph)
        self.assertTrue(np.all(np.diag(jac_simple) == 1.0))

        _, _, _, jac_no_edges = calculate_jaccard_similarity(self.no_edges_graph)
        self.assertTrue(np.all(np.diag(jac_no_edges) == 1.0))

if __name__ == '__main__':
    unittest.main()