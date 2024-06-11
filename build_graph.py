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
substring_to_remove_links = np.array(['Wikipedia', 'Category', 'identifier', 'Help', 'Template', ':', 'Wayback', 'ISBN'])
substring_to_remove_categories = np.array(['Wiki', 'CS1', 'Articles', 'Pages', 'redirects', 'template', 'disputes', 'ISO'])

wiki_wiki = wikipediaapi.Wikipedia('Missing Knowledge Links', 'en')

def fetch_links(start_pages):
    # URL-encode each page title
    encoded_titles = [urllib.parse.quote(title) for title in start_pages]
    # Join the encoded titles with the pipe character
    titles = "|".join(encoded_titles)
    url = f'https://en.wikipedia.org/w/api.php?action=query&titles={titles}&prop=links&pllimit=max&format=json'
    print('url: ', url)
    response = requests.get(url)
    data = response.json()

    print(data)
    all_links = {}

    pages = data['query']['pages']
    print('Pages: ', pages)
    
    for page_id in pages:
        print('')
        print('Page ID: ', page_id)
        page_title = str(pages[page_id]['title'])
        print('Page Title: ', page_title)

        if 'links' in pages[page_id]:
            print('Links: ', pages[page_id]['links'])

            # make a dataframe from the links
            links_df = pd.DataFrame(pages[page_id]['links'])

            print('Links DF: ', links_df)
            
            # take the title column into an array
            links = np.array(links_df['title'], dtype=str)

            # create a mask
            mask = np.zeros(len(links), dtype=bool)

            # iterate over the blacklist substrings
            for substring in blacklist_substrings:
                current_mask = np.char.find(links, substring) != -1
                mask |= current_mask

            # Filter the array using the combined mask
            filtered_array = links[~mask]

            # add the filtered array to the links dictionary
            print('page_title: ', page_title)

            all_links[page_title] = filtered_array

    return all_links


def fetch_links_api(page_title, get_categories=True):

    page = wiki_wiki.page(page_title)
    links = page.links
    
    # make an array of the keys of the links dictionary
    link_titles = np.array(list(links.keys()), dtype=str)

    # create a mask
    link_mask = np.zeros(len(link_titles), dtype=bool)

    # Create a 2D array where each row corresponds to a link and each column to a substring check
    contains_substring = np.array([np.char.find(link_titles, substring) != -1 for substring in substring_to_remove_links])

    # Combine the results across all substrings to create a final mask
    link_mask = np.any(contains_substring, axis=0)

    # Filter the array using the combined mask
    filtered_links_array = link_titles[~link_mask]

    if not get_categories:
        return filtered_links_array
    
    categories = page.categories

    # make an array of the keys of the categories dictionary
    category_titles = np.array(list(categories.keys()), dtype=str)

    # remove 'Category:' from the category titles
    category_titles = np.char.replace(category_titles, 'Category:', '')

    # create a 2D array where each row corresponds to a category and each column to a substring check
    contains_substring = np.array([np.char.find(category_titles, substring) != -1 for substring in substring_to_remove_categories])

    # Combine the results across all substrings to create a final mask
    category_mask = np.any(contains_substring, axis=0)

    # Filter the array using the combined mask
    filtered_categories_array = category_titles[category_mask]

    return filtered_links_array, filtered_categories_array


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


def complete_graph(G, links_dict, min_links=15):

    # List all the nodes in the graph
    nodes = list(G.nodes)

    def process_node(node):
        # Check if the node is in the links dictionary
        if node not in links_dict:
            # Fetch the links for the node
            links = fetch_links_api(node, get_categories=False)
            # Add the links to the links dictionary
            links_dict[node] = links

    # Use ThreadPoolExecutor to multithread the processing of nodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        executor.map(process_node, nodes)

    for node in nodes:

        added_links = []

        # get the links for the node
        if node in links_dict:
            links = links_dict[node]

        else:
            links = fetch_links_api(node, get_categories=False)

        # for link in links
        for link in links:

            # check if the link is in the graph but the edge is not
            if link in G and not G.has_edge(node, link):

                # add the edge
                G.add_edge(node, link)


                # add the link to the added links
                added_links.append(link)

    # remove nodes that have less than min_links
    nodes_to_remove = [node for node in G.nodes if G.degree(node) < min_links]

    for node in nodes_to_remove:
        G.remove_node(node)

    print('Graph completed with new links between already present nodes.')
    print('Number of nodes:', G.number_of_nodes(), 'Number of edges:', G.number_of_edges())

    return G, links_dict

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
        print('Number of nodes:', completed_graph.number_of_nodes(), 'Number of edges:', completed_graph.number_of_edges())
    except:

        print('Graph not found. Building a new graph.')
        # Build the initial graph
        graph, links_dict, categories_dict = build_wikipedia_graph(start_page, depth)

        # Complete the graph
        completed_graph, links_dict = complete_graph(graph, links_dict)

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
