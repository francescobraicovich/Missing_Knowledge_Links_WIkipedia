"""
Module for building and processing Wikipedia article graphs.

This module provides functions to:
- Fetch links and categories from Wikipedia pages.
- Build a graph structure based on these links, exploring to a specified depth.
- Complete the graph by adding links between already existing nodes.
- Filter links and categories based on predefined substrings.
- Save and load graph data (graph structure, links, categories) to/from files.
"""

import networkx as nx
import requests # wikipediaapi might use it; also for potential future direct HTTP tasks
import numpy as np
import urllib.parse # wikipediaapi might use it
import wikipediaapi
import os # Needed for ThreadPoolExecutor default max_workers calculation (os.cpu_count())
import pickle
import concurrent.futures
import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

# Define the blacklist substrings
substring_to_remove_links = ['wikipedia', 'category', 'identifier', 'help', 'template', ':', 'wayback', 'isbn', 'jstor']
substring_to_remove_categories = ['wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes', 'iso', 'dmy', 'short', 's1']

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MissingLinksAnalyzer/1.0 (https://example.com/missinglinksanalyzer; cooldev@example.com)', # Added example user-agent
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


def _filter_sentences_vectorised(sentences: np.ndarray, words_to_filter: np.ndarray) -> np.ndarray:
    """
    Filters out sentences that contain any of the specified words to filter (vectorized version).

    Args:
        sentences (np.ndarray): An array of sentences to filter.
        words_to_filter (np.ndarray): An array of words to filter.

    Returns:
        np.ndarray: An array of filtered sentences.
    """
    if len(sentences) == 0:
        return sentences

    # convert all sentences to lowercase
    sentences_lower = np.char.lower(sentences) # Changed variable name for clarity

    # find the number of spaces in each sentence
    n_spaces = np.char.count(sentences_lower, ' ')

    # find the maximum number of spaces
    n_words_to_add = np.max(n_spaces) - n_spaces
    words_to_add = np.char.multiply(' 0', n_words_to_add) # Note: leading space in ' 0'

    # add the words to the sentences
    equal_length_sentences = np.char.add(sentences_lower, words_to_add)
    
    # split the sentences into words
    words_array_list = np.char.split(equal_length_sentences) # Renamed for clarity
    
    # stack the words into a 2D array
    # This part is tricky as np.vstack on a list of lists of different lengths won't work directly
    # Assuming sentences are padded to have the same number of "words" after split
    # If not, this needs a more robust way to form a 2D array or a different approach
    try:
        words_array = np.array(words_array_list.tolist()) # Attempt to make it a regular list of lists for np.array
    except AttributeError: # if already a list (e.g. from older numpy)
        words_array = np.array(words_array_list)


    # create a filter mask
    filter_mask_2d = np.isin(words_array, words_to_filter)

    # find the rows that contain at least one word to filter
    filter_mask = np.any(filter_mask_2d, axis=1)

    # filter the sentences (use original sentences, not the lowercased or padded ones)
    filtered_sentences = sentences[~filter_mask]

    return filtered_sentences


def _filter_sentences(sentences: np.ndarray, words_to_filter: list[str]) -> np.ndarray:
    """
    Filters out sentences that contain any of the specified words to filter.

    Args:
        sentences (np.ndarray): An array of sentences to filter.
        words_to_filter (list[str]): A list of words to filter.

    Returns:
        np.ndarray: An array of filtered sentences.
    """
    if sentences.size == 0:
        return sentences
    # convert the sentences to lowercase
    sentences_lower = np.char.lower(sentences)

    # create a mask 
    mask = np.zeros(len(sentences_lower), dtype=bool)
        
    # iterate over the words to filter
    for word in words_to_filter:
        # Ensure word is also lowercase for consistent comparison
        current_mask = np.char.find(sentences_lower, word.lower()) != -1
        mask |= current_mask

    # filter the sentences
    filtered_sentences = sentences[~mask]

    return filtered_sentences


def _fetch_links_api(page_title: str, filter_links: bool = True, filter_categories: bool = True, get_categories: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Fetches the links and categories of a Wikipedia page using wikipediaapi.

    Args:
        page_title (str): The title of the Wikipedia page.
        filter_links (bool, optional): Whether to filter the links. Defaults to True.
        filter_categories (bool, optional): Whether to filter the categories. Defaults to True.
        get_categories (bool, optional): Whether to retrieve the categories. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: A tuple containing an array of link titles,
                                              and an array of category titles or None if get_categories is False.
                                              Returns (empty_array, empty_array) or (empty_array, None) on error.
    """
    try:
        page = wiki_wiki.page(page_title)
        if not page.exists():
            logger.warning(f"Page '{page_title}' does not exist.")
            return np.array([], dtype=str), (np.array([], dtype=str) if get_categories else None)

        links = page.links
        link_titles = np.array(list(links.keys()), dtype=str) # Use .keys() for consistency

        if filter_links:
            filtered_links_array = _filter_sentences(link_titles, substring_to_remove_links)
        else:
            filtered_links_array = link_titles

        if not get_categories:
            return filtered_links_array, None

        categories = page.categories
        category_titles = np.array(list(categories.keys()), dtype=str)
        category_titles = np.char.lstrip(category_titles, 'Category:') # lstrip works on string arrays

        if filter_categories:
            filtered_categories_array = _filter_sentences(category_titles, substring_to_remove_categories)
        else:
            filtered_categories_array = category_titles
        
        return filtered_links_array, filtered_categories_array

    except requests.exceptions.RequestException as e: # More specific exception
        logger.error(f"Network error fetching page '{page_title}': {e}")
        return np.array([], dtype=str), (np.array([], dtype=str) if get_categories else None)
    except Exception as e:
        logger.error(f"Error fetching links for page '{page_title}': {e}")
        return np.array([], dtype=str), (np.array([], dtype=str) if get_categories else None)


def _build_wikipedia_graph(start_page: str, depth: int) -> tuple[nx.Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Builds an initial Wikipedia graph by crawling links from a starting page up to a specified depth.

    Args:
        start_page (str): The title of the starting Wikipedia page.
        depth (int): The depth of the graph crawl.

    Returns:
        tuple[nx.Graph, dict[str, np.ndarray], dict[str, np.ndarray]]: 
            A tuple containing the built graph, a dictionary of links for each page,
            and a dictionary of categories for each page.
    """
    G = nx.Graph()
    G.add_node(start_page)

    to_visit: dict[int, list[str]] = {i: [] for i in range(depth + 1)}
    to_visit[0] = [start_page]
    
    visited: set[str] = {start_page} # Initialize with start_page
    links_dict: dict[str, np.ndarray] = {}
    categories_dict: dict[str, np.ndarray] = {} # Ensure categories_dict is always a dict of np.ndarray
    
    for i in range(depth):
        logger.info(f"Building graph at depth {i}, {len(to_visit[i])} pages to visit.")
        current_depth_pages = list(to_visit[i]) # Iterate over a copy

        for current_page_title in current_depth_pages: # Use a more descriptive name
            if current_page_title not in links_dict: # Process only if not already fetched
                links, categories = _fetch_links_api(current_page_title)
                
                links_dict[current_page_title] = links
                if categories is not None: # Ensure categories is not None
                    categories_dict[current_page_title] = categories
                else: # Should not happen if get_categories is True by default in _fetch_links_api for this path
                    categories_dict[current_page_title] = np.array([], dtype=str)


                for link_title in links: # Use a more descriptive name
                    if link_title not in G:
                        G.add_node(link_title)
                    G.add_edge(current_page_title, link_title)

                    if link_title not in visited:
                        if i + 1 <= depth: # Check depth boundary
                            to_visit[i + 1].append(link_title)
                        visited.add(link_title) # Add to visited when enqueued or processed to avoid re-processing
            
            # Ensure current_page_title is added to visited even if it was already in links_dict
            # (e.g. if it was added as a link from another page but not yet processed itself)
            visited.add(current_page_title)


    logger.info('Initial graph built.')
    logger.info(f'Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}')
    logger.info(f'Number of pages with category data: {len(categories_dict)}')
    return G, links_dict, categories_dict


def _complete_graph(G: nx.Graph, links_dict: dict[str, np.ndarray], categories_dict: dict[str, np.ndarray], min_links: int) -> tuple[nx.Graph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Completes the graph by adding missing links between existing nodes and pruning nodes with few links.

    Args:
        G (nx.Graph): The graph to complete.
        links_dict (dict[str, np.ndarray]): A dictionary containing the links for each node.
        categories_dict (dict[str, np.ndarray]): A dictionary containing the categories for each node.
        min_links (int): The minimum number of links a node must have to be kept in the graph.

    Returns:
        tuple[nx.Graph, dict[str, np.ndarray], dict[str, np.ndarray]]: 
            A tuple containing the completed graph, updated links dictionary, 
            and updated categories dictionary.
    """
    nodes = list(G.nodes)
    logger.info(f"Completing graph with {len(nodes)} nodes. Fetching any missing link data.")

    def process_node(node_title: str): # Use a more descriptive name
        if node_title not in links_dict: # Only fetch if not already present
            # When completing graph, we want all links/categories, so filter_links/categories = False
            links, categories = _fetch_links_api(node_title, filter_links=False, filter_categories=False)
            links_dict[node_title] = links
            if categories is not None:
                categories_dict[node_title] = categories
            else: # Should not happen
                categories_dict[node_title] = np.array([], dtype=str)


    # Use ThreadPoolExecutor to multithread the processing of nodes
    # os.cpu_count() could be an option for max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # Filter nodes that are not in links_dict to avoid redundant processing
        nodes_to_fetch = [node for node in nodes if node not in links_dict]
        if nodes_to_fetch:
             logger.info(f"Fetching link data for {len(nodes_to_fetch)} nodes during completion.")
             executor.map(process_node, nodes_to_fetch)
        else:
            logger.info("No missing link data to fetch during completion.")


    logger.info("Adding missing edges between existing nodes.")
    for node_title in nodes: # Use a more descriptive name
        # Node should be in links_dict now due to process_node call or initial build
        if node_title in links_dict:
            node_links = links_dict[node_title] # Use a more descriptive name
            for link_title in node_links: # Use a more descriptive name
                if link_title in G and not G.has_edge(node_title, link_title):
                    G.add_edge(node_title, link_title)
        else:
            logger.warning(f"Node '{node_title}' still not in links_dict after fetching attempt. Skipping its edge completion.")


    # Pruning nodes
    if G.number_of_nodes() > 0: # Avoid error with empty graph
        degrees = G.degree()
        # Quantile calculation might fail if all degrees are the same or too few nodes
        try:
            # Ensure there are enough nodes to calculate quantiles if that logic is re-introduced
            # For now, directly use min_links from params
            pass # quantile_033 = np.quantile([d for n, d in degrees], 0.33) 
                 # min_links_effective = int(min(min_links, quantile_033))
        except Exception as e:
            logger.warning(f"Could not calculate degree quantile for min_links adjustment: {e}. Using provided min_links: {min_links}")
            # min_links_effective = min_links
        
        min_links_effective = min_links # Using the parameter directly as per refactor plan

        nodes_to_remove = [node for node, degree_val in degrees if degree_val < min_links_effective] # Corrected iteration
        
        if nodes_to_remove:
            logger.info(f"Removing {len(nodes_to_remove)} nodes with degree < {min_links_effective}.")
            for node_to_remove in nodes_to_remove: # Use a more descriptive name
                G.remove_node(node_to_remove)
                # Also remove from links_dict and categories_dict if desired, though not strictly necessary
                # if they are only ever accessed via G.nodes()
                links_dict.pop(node_to_remove, None)
                categories_dict.pop(node_to_remove, None)
        else:
            logger.info(f"No nodes to remove based on min_links criterion (degree < {min_links_effective}).")
    else:
        logger.info("Graph is empty, skipping pruning.")


    logger.info('Graph completed.')
    logger.info(f'Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}')
    logger.info(f'Number of pages with category data: {len(categories_dict)}')
    return G, links_dict, categories_dict


def build_graph(start_page: str, depth: int, min_links_completion: int) -> tuple[nx.Graph | None, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """
    Builds or loads a Wikipedia graph for a given start page and depth.

    The process involves:
    1. Constructing paths for graph data based on settings and input parameters.
    2. Attempting to load an existing graph and associated data (links, categories) from these paths.
    3. If not found, it builds a new graph:
        a.  Calls `_build_wikipedia_graph` to create an initial graph by crawling from the start_page.
        b.  Calls `_complete_graph` to add more links between existing nodes and prune sparse nodes.
        c.  Saves the newly built graph and its associated data.
    
    Args:
        start_page (str): The title of the starting Wikipedia page.
        depth (int): The depth for the graph crawl (how many links away from start_page to explore).
        min_links_completion (int): The minimum number of links a node must have after the 
                                    completion phase to be retained in the graph.

    Returns:
        tuple[nx.Graph | None, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]: 
            A tuple containing the graph, links dictionary, and categories dictionary.
            Returns (None, None, None) if a critical error occurs during graph building (e.g. start page not found).
    """
    # Sanitize start_page for directory naming (replace spaces, etc.)
    safe_start_page_name = start_page.replace(" ", "_").replace("/", "_") # Basic sanitization
    folder_name = f'{safe_start_page_name}_(Depth_{depth})' # Consistent naming
    folder_path = settings.paths.graph_output_dir / folder_name
    
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception(f"Could not create directory {folder_path}: {e}. Check permissions and path configuration.")
        return None, None, None


    graph_path = folder_path / 'graph.gexf'
    links_path = folder_path / 'links.pkl'
    categories_path = folder_path / 'categories.pkl'

    try: 
        logger.info(f"Attempting to load existing graph data from {folder_path}")
        loaded_graph = nx.read_gexf(graph_path)
        
        with open(links_path, 'rb') as f_links, open(categories_path, 'rb') as f_cats:
            loaded_links_dict = pickle.load(f_links)
            loaded_categories_dict = pickle.load(f_cats)

        logger.info(f"Graph loaded successfully from {graph_path}.")
        logger.info(f"Number of nodes: {loaded_graph.number_of_nodes()}, Number of edges: {loaded_graph.number_of_edges()}")
        return loaded_graph, loaded_links_dict, loaded_categories_dict
    
    except FileNotFoundError:
        logger.info(f"No existing graph data found at {folder_path}. Building a new graph for '{start_page}'.")
    except (pickle.UnpicklingError, IOError, nx.NetworkXError) as e: # More specific exceptions for loading
        logger.exception(f"Error loading existing graph data from {folder_path}: {e}. Will attempt to build a new graph.")
    except Exception as e: # Catch any other unexpected errors during loading
        logger.exception(f"Unexpected error loading existing graph data from {folder_path}: {e}. Will attempt to build a new graph.")


    # If loading failed or file not found, build a new one
    # Check if start page exists before building
    try:
        start_page_check = wiki_wiki.page(start_page)
        if not start_page_check.exists():
            logger.error(f"The start page '{start_page}' does not exist on Wikipedia. Cannot build graph.")
            return None, None, None # Indicate failure
    except requests.exceptions.RequestException as e:
        logger.exception(f"Network error while checking if start page '{start_page}' exists: {e}")
        return None, None, None


    logger.info(f"Building initial graph for '{start_page}' with depth {depth}.")
    # Build the initial graph
    graph, links_dict, categories_dict = _build_wikipedia_graph(start_page, depth)

    if graph.number_of_nodes() == 0 : 
        logger.warning(f"Initial graph for '{start_page}' resulted in zero nodes. Check start page or depth.")
        # Proceed to save this empty state if that's desired, or handle differently.
    
    logger.info("Completing the graph with missing links and pruning.")
    completed_graph, completed_links_dict, completed_categories_dict = _complete_graph(
        graph, links_dict, categories_dict, min_links=min_links_completion
    )

    try:
        logger.info(f"Saving new graph data to {folder_path}")
        nx.write_gexf(completed_graph, graph_path)
        with open(categories_path, 'wb') as f_cats:
            pickle.dump(completed_categories_dict, f_cats)
        with open(links_path, 'wb') as f_links:
            pickle.dump(completed_links_dict, f_links)
        logger.info("Graph data saved successfully.")
    except (IOError, pickle.PicklingError, nx.NetworkXError) as e: # More specific exceptions for saving
        logger.exception(f"Error saving graph data to {folder_path}: {e}")
        # Decide if to return the in-memory graph or None. Returning in-memory version for now.
        return completed_graph, completed_links_dict, completed_categories_dict 
    except Exception as e: # Catch any other unexpected errors during saving
        logger.exception(f"Unexpected error saving graph data to {folder_path}: {e}")
        return completed_graph, completed_links_dict, completed_categories_dict


    return completed_graph, completed_links_dict, completed_categories_dict

# Removed matplotlib.pyplot as display functionality is removed.
# `os` import kept as it's implicitly used by ThreadPoolExecutor's default for max_workers.
# `requests` and `urllib.parse` kept as `wikipediaapi` might rely on them being available.
# Removed if __name__ == "__main__": block.
