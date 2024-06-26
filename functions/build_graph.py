import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse
import wikipediaapi
import os
import pickle as pkl
import concurrent.futures

# Define the blacklist substrings
substring_to_remove_links = ['wikipedia', 'category', 'identifier', 'help', 'template', ':', 'wayback', 'isbn', 'jstor']
substring_to_remove_categories = ['wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes', 'iso', 'dmy', 'short', 's1']

wiki_wiki = wikipediaapi.Wikipedia('Missing Knowledge Links', 'en')


def filter_sentences_vectorised(sentences, words_to_filter):
    """
    Filters out sentences that contain any of the specified words to filter.

    Args:
        sentences (numpy.ndarray): An array of sentences to filter.
        words_to_filter (numpy.ndarray): An array of words to filter.

    Returns:
        numpy.ndarray: An array of filtered sentences.

    """
    if len(sentences) == 0:
        return sentences

    # convert all sentences to lowercase
    sentences = np.char.lower(sentences)

    # find the number of spaces in each sentence
    n_spaces = np.char.count(sentences, ' ')

    # find the maximum number of spaces
    n_words_to_add = np.max(n_spaces) - n_spaces
    words_to_add = np.char.multiply(' 0', n_words_to_add)

    # add the words to the sentences
    equal_length_sentences = np.char.add(sentences, words_to_add)
    
    # split the sentences into words
    words_array = np.char.split(equal_length_sentences)
    
    # stack the words into a 2D array
    words_array = np.vstack(words_array)

    # create a filter mask
    filter_mask_2d = np.isin(words_array, words_to_filter)

    # find the rows that contain at least one word to filter
    filter_mask = np.any(filter_mask_2d, axis=1)

    # filter the sentences
    filtered_sentences = sentences[~filter_mask]

    return filtered_sentences


def filter_sentences(sentences, words_to_filter):
    """
    Filters out sentences that contain any of the specified words to filter.

    Args:
        sentences (numpy.ndarray): An array of sentences to filter.
        words_to_filter (list): A list of words to filter.

    Returns:
        numpy.ndarray: An array of filtered sentences.
    """
    # convert the sentences to lowercase
    sentences_lower = np.char.lower(sentences)

    # create a mask 
    mask = np.zeros(len(sentences_lower), dtype=bool)
        
    # iterate over the words to filter
    for word in words_to_filter:
        current_mask = np.char.find(sentences_lower, word) != -1
        mask |= current_mask

    # filter the sentences
    filtered_sentences = sentences[~mask]

    return filtered_sentences


def fetch_links_api(page_title, filter_links=True, filter_categories=True, get_categories=True, retry=0):
    """
    Fetches the links and categories of a Wikipedia page.

    Args:
        page_title (str): The title of the Wikipedia page.
        filter_links (bool, optional): Whether to filter the links. Defaults to True.
        filter_categories (bool, optional): Whether to filter the categories. Defaults to True.
        get_categories (bool, optional): Whether to retrieve the categories. Defaults to True.
        retry (int, optional): The number of times to retry the API call in case of failure. Defaults to 0.

    Returns:
        tuple: A tuple containing the filtered links array and the filtered categories set.
    """
    page = wiki_wiki.page(page_title)
    links = page.links

    # extract the links as an array
    link_titles = np.array(list(links), dtype=str)

    if filter_links:
        # filter the links
        filtered_links_array = filter_sentences(link_titles, substring_to_remove_links)
    else:
        filtered_links_array = link_titles

    if not get_categories:
        return filtered_links_array

    categories = page.categories

    # make an array of the keys of the categories dictionary
    category_titles = np.array(list(categories.keys()), dtype=str)

    # remove the first 9 characters of the category titles
    category_titles = np.char.lstrip(category_titles, 'Category:')

    if filter_categories:
        # filter the categories
        filtered_categories_set = filter_sentences(category_titles, substring_to_remove_categories)
    else:
        filtered_categories_set = category_titles

    return filtered_links_array, filtered_categories_set


def build_wikipedia_graph(start_page, depth, verbosity=0):
    """
    Builds a Wikipedia graph starting from a given page.

    Args:
        start_page (str): The title of the starting Wikipedia page.
        depth (int): The depth of the graph.
        verbosity (int, optional): The level of verbosity. Defaults to 0.

    Returns:
        tuple: A tuple containing the built graph, links dictionary, and categories dictionary.
    """
    # Create an empty graph
    G = nx.Graph()
    # Add the starting page as a node in the graph
    G.add_node(start_page)

    # Create a dictionary to store the pages to visit at each depth
    to_visit = {0: [start_page]}

    # Create a list in the dictionary for each depth
    for i in range(1, depth + 1):
        to_visit[i] = []
    
    # Create a set to store the visited pages
    visited = set(start_page)

    # Create dictionaries to store the links and categories
    links_dict = {}
    categories_dict = {}
    
    # Iterate over the depths
    for i in range(depth):

        # Get the pages to visit at the current depth
        to_visit_at_depth = to_visit[i]

        # Iterate over the pages to visit at the current depth
        while to_visit_at_depth:
            # Get the current page
            current_page = to_visit_at_depth.pop(0)
            
            # Fetch the links and categories for the current page
            links, categories = fetch_links_api(current_page)

            # Add the links and categories to the dictionaries
            links_dict[current_page] = links
            categories_dict[current_page] = categories

            # Iterate over the links
            for link in links:

                # Check if the link is not already in the graph
                if link not in G:
                    # Add the link as a node in the graph
                    G.add_node(link)

                # Add an edge between the current page and the link
                G.add_edge(current_page, link)

                # Check if the link has not been visited
                if link not in visited:
                    # Add the link to the pages to visit at the next depth
                    to_visit[i + 1].append(link)

                # Add the current page to the visited set
                visited.add(current_page)

    # Print the graph information if verbosity is greater than 0
    if verbosity > 0:
        print('Initial graph built.')
        print('Number of nodes:', G.number_of_nodes(), 'Number of edges:', G.number_of_edges())
        print('Number of categories:', len(categories_dict))
        print('')

    return G, links_dict, categories_dict


def complete_graph(G, links_dict, categories_dict, min_links=15):
    """
    Completes the graph by adding missing links between existing nodes.

    Args:
        G (networkx.Graph): The graph to complete.
        links_dict (dict): A dictionary containing the links for each node.
        categories_dict (dict): A dictionary containing the categories for each node.
        min_links (int, optional): The minimum number of links a node must have to be kept in the graph. Defaults to 15.

    Returns:
        tuple: A tuple containing the completed graph, updated links dictionary, and updated categories dictionary.
    """

    # List all the nodes in the graph
    nodes = list(G.nodes)

    print('First round: Processing nodes to add missing links between existing nodes.')

    def process_node(node):
        """
        Fetches the links and categories for a given node and updates the links and categories dictionaries.

        Args:
            node (str): The node to process.
        """
        # Check if the node is in the links dictionary
        if node not in links_dict:
            # Fetch the links for the node
            links, categories = fetch_links_api(node, filter_links=False, filter_categories=False)
    
            # Add the links to the links dictionary
            links_dict[node] = links

            # Add the categories to the categories dictionary
            categories_dict[node] = categories


    # Use ThreadPoolExecutor to multithread the processing of nodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        executor.map(process_node, nodes)

    for node in links_dict.keys():

        # Get the links of the node
        links = links_dict[node]
        
        # for link in links
        for link in links:

            # check if the link is in the graph but the edge is not
            if link in G and not G.has_edge(node, link):
                # add the edge
                G.add_edge(node, link)

    # find the degree of each node
    degrees = G.degree

    # find the 0.33 quantile of the degrees
    quantile_033 = np.quantile(list(dict(degrees).values()), 0.33)

    min_links = int(min(min_links, quantile_033))   

    # remove nodes that have less than min_links
    nodes_to_remove = [node for node in G.nodes if G.degree(node) < min_links]

    for node in nodes_to_remove:
        G.remove_node(node)
        pass

    print('Graph completed with new links between already existing nodes.')
    print('Number of nodes:', G.number_of_nodes(), ', Number of edges:', G.number_of_edges(), ', Number of categories:', len(categories_dict))

    return G, links_dict, categories_dict



def build_graph(start_page, depth, verbosity=0, display=False):
    """
    Builds a Wikipedia graph starting from a given page.

    Args:
        start_page (str): The title of the starting Wikipedia page.
        depth (int): The depth of the graph.
        verbosity (int, optional): The level of verbosity. Defaults to 0.
        display (bool, optional): Whether to display the graph. Defaults to False.

    Returns:
        tuple: A tuple containing the built graph, links dictionary, and categories dictionary.
    """
    # Define the folder path to save the graph, links, and categories
    folder_path = f'graphs/{start_page}_(Depth: {depth}'
    # create a new folder for the graph if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    graph_path = folder_path + '/graph.gexf' # Change this to the path you want to save the graph to
    links_path = folder_path + '/links.pkl' # Change this to the path you want to save the links to
    categories_path = folder_path + '/categories.pkl' # Change this to the path you want to save the categories to

    try: 
        # Try to open the existing graph 
        completed_graph = nx.read_gexf(graph_path)
        
        # Load pickled links and categories
        with open(links_path, 'rb') as f:
            links_dict = pkl.load(f)

        with open(categories_path, 'rb') as f:
            categories_dict = pkl.load(f)

        print('Graph found. Loading graph, links, and categories.')
        print('Number of nodes:', completed_graph.number_of_nodes(), ', Number of edges:', completed_graph.number_of_edges())
    except:
        # If the graph doesn't exist, build a new one
        print('Graph not found. Building a new graph.')
        # Build the initial graph
        graph, links_dict, categories_dict = build_wikipedia_graph(start_page, depth)

        print('Graph built. Completing the graph with missing links between existing nodes.')
        # Complete the graph
        completed_graph, links_dict, categories_dict = complete_graph(graph, links_dict, categories_dict)

        # Save the completed graph as a GEXF file
        nx.write_gexf(completed_graph, graph_path)

        # Save the links and categories as pickled files
        with open(categories_path, 'wb') as f:
            pkl.dump(categories_dict, f)
        
        with open(links_path, 'wb') as f:
            pkl.dump(links_dict, f)
    
    if display:
        # Draw the graph
        plt.figure(figsize=(12, 12))
        nx.draw(completed_graph, with_labels=True, font_size=8, node_size=100)
        plt.show()

    return completed_graph, links_dict, categories_dict
