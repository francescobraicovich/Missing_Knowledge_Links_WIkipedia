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


def fetch_links_api(page_title, filter_links=True, filter_categories = True, get_categories=True, retry=0):

    page = wiki_wiki.page(page_title)
    links = page.links

    # extract the links as an array
    link_titles = np.array(list(links), dtype=str)

    
    if filter_links:
        # filter the links
        filtered_links_array = filter_sentences(link_titles, substring_to_remove_links)
    else:
        filtered_links_array = link_titles

    if not get_categories:
        return filtered_links_array
    
    categories = page.categories

    # make an array of the keys of the categories dictionary
    category_titles = np.array(list(categories.keys()), dtype=str)

    # remove the first 9 characters of the category titles
    category_titles = np.char.lstrip(category_titles, 'Category:')

    if filter_categories:
        # filter the categories
        filtered_categories_set = filter_sentences(category_titles, substring_to_remove_categories)
    else:
        filtered_categories_set = category_titles

    return filtered_links_array, filtered_categories_set


def build_wikipedia_graph(start_page, depth, verbosity=0):
    """Build an undirected graph of Wikipedia pages up to a certain depth."""
    G = nx.Graph()
    G.add_node(start_page)

    to_visit = {0: [start_page]}

    # create a list in the dictionary for each depth
    for i in range(1, depth + 1):
        to_visit[i] = []
    
    # create a set of visited pages
    visited = set(start_page)

    # create a dictionary to store the links and categories
    links_dict = {}
    categories_dict = {}
    
    for i in range(depth):

        to_visit_at_depth = to_visit[i]

        for current_page in to_visit_at_depth:

            current_page = to_visit_at_depth.pop(0)
            
            links, categories = fetch_links_api(current_page)

            links_dict[current_page] = links
            categories_dict[current_page] = categories

            for link in links:

                # check if the link is in the graph
                if link not in G:
                    G.add_node(link)

                # add an edge between the page and the link
                G.add_edge(current_page, link)

                if link not in visited:
                    to_visit[i + 1].append(link)

                # add the page to the visited set
                visited.add(current_page)

    print('Initial graph built.')
    print('Number of nodes:', G.number_of_nodes(), 'Number of edges:', G.number_of_edges())
    print('Number of categories:', len(categories_dict))
    print('')

    return G, links_dict, categories_dict


def complete_graph(G, links_dict, categories_dict, min_links=15):

    # List all the nodes in the graph
    nodes = list(G.nodes)

    print('First round: Processing nodes to add missing links between existing nodes.')

    def process_node(node):
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

    for node in nodes:

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

    folder_path = f'graphs/{start_page}_(Depth: {depth})'
    # create a new folder for the graph
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    graph_path = folder_path + '/graph.gexf' # Change this to the path you want to save the graph to
    links_path = folder_path + '/links.pkl' # Change this to the path you want to save the links to
    categories_path = folder_path + '/categories.pkl' # Change this to the path you want to save the categories to

    try: 
        # open graph 
        completed_graph = nx.read_gexf(graph_path)
        
        # load pickled links and categories
        with open(links_path, 'rb') as f:
            links_dict = pkl.load(f)

        with open(categories_path, 'rb') as f:
            categories_dict = pkl.load(f)

        print('Graph found. Loading graph, links and categories.')
        print('Number of nodes:', completed_graph.number_of_nodes(), ', Number of edges:', completed_graph.number_of_edges())
    except:

        print('Graph not found. Building a new graph.')
        # Build the initial graph
        graph, links_dict, categories_dict = build_wikipedia_graph(start_page, depth)


        print('Graph built. Completing the graph with missing links between existing nodes.')
        # Complete the graph
        completed_graph, links_dict, categories_dict = complete_graph(graph, links_dict, categories_dict)

        nx.write_gexf(completed_graph, graph_path)

        # dump with pickle the links and categories
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
