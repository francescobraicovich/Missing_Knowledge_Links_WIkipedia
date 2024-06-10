import wikipediaapi
import networkx as nx
import asyncio
import aiohttp
from collections import deque
import nest_asyncio
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from categorize_graph import categorize_graph



# Define the blacklist substrings
blacklist_substrings = {'Wikipedia', 'Category', 'identifier', 'Help', 'Template', ':', 'Wayback'}  # Add actual blacklisted substrings here

def is_blacklisted(page_name, blacklist_substrings):
    return any(substring in page_name for substring in blacklist_substrings)

def get_wikipedia_links(session, title):
    """Fetches the links for a given Wikipedia page title using the Wikipedia API."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'links',
        'titles': title,
        'pllimit': 'max'  # Get maximum number of links
    }
    response = session.get(url, params=params)
    data = response.json()
    pages = data['query']['pages']
    links = []

    for page_id in pages:
        if 'links' in pages[page_id]:
            links.extend([link['title'] for link in pages[page_id]['links']])

    return links

def build_wikipedia_graph(start_page, depth, verbosity):
    """Builds a Wikipedia graph starting from a given page using BFS up to a specified depth."""
    # Initialize the graph
    G = nx.Graph()
    
    # Cache to store fetched pages
    page_cache = {}
    
    # Queue for BFS
    queue = deque([(start_page, 0)])
    visited = set()

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=None) as executor:
            while queue:
                current_batch = []
                while queue and len(current_batch) < 10:
                    current_page, current_depth = queue.popleft()
                    if current_page not in visited and current_depth < depth:
                        visited.add(current_page)
                        current_batch.append((current_page, current_depth))
                
                tasks = {executor.submit(get_wikipedia_links, session, page): (page, current_depth) for page, current_depth in current_batch}
                for future in as_completed(tasks):
                    page, current_depth = tasks[future]
                    try:
                        links = future.result()
                        page_cache[page] = links
                        
                        if not links:
                            continue
                        
                        # Check if the page name contains any substring from the blacklist
                        if is_blacklisted(page, blacklist_substrings):
                            continue
                        
                        # Add the page to the graph
                        G.add_node(page)
                        
                        # Add links to the graph
                        for link in links:
                            if is_blacklisted(link, blacklist_substrings):
                                continue
                            G.add_edge(page, link)
                            if link not in visited:
                                queue.append((link, current_depth + 1))
                    except Exception as exc:
                        print(f'Error fetching links for {page}: {exc}')
    
    if verbosity >= 1:
        print('')
        print('-'*50, flush=True)
        print('First Graph built', flush=True)
        print('Number of nodes:', G.number_of_nodes(), 'Number of edges:', G.number_of_edges(), flush=True)
        print('')
   
    return G

from threading import Lock

def get_wikipedia_links(session, title):
    """Fetches the links for a given Wikipedia page title using the Wikipedia API."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'links',
        'titles': title,
        'pllimit': 'max'  # Get maximum number of links
    }
    response = session.get(url, params=params)
    data = response.json()
    pages = data['query']['pages']
    links = []

    for page_id in pages:
        if 'links' in pages[page_id]:
            links.extend([link['title'] for link in pages[page_id]['links']])

    return links

def add_edges_to_graph(graph, title, links, lock):
    """Adds edges to the graph for existing nodes in a thread-safe manner."""
    with lock:
        for link in links:
            if link in graph.nodes and not graph.has_edge(title, link):
                graph.add_edge(title, link)

def finish_graph(graph):
    """Fetches links from each page in the graph and adds missing edges among existing nodes."""
    # Extract the page titles from the graph nodes
    page_titles = list(graph.nodes)
    
    # Use a dictionary to hold the links for each page
    links_dict = {}
    
    # Use a session for connection pooling
    session = requests.Session()
    
    # Fetch links in parallel with increased workers
    with ThreadPoolExecutor(max_workers=None) as executor:  # Increase the number of workers for more parallelism
        future_to_title = {executor.submit(get_wikipedia_links, session, title): title for title in page_titles}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                links = future.result()
                links_dict[title] = links
            except Exception as exc:
                print(f'Error fetching links for {title}: {exc}')

    print('Links fetched', flush=True)
    
    # Add missing edges among existing nodes in parallel
    lock = Lock()
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(add_edges_to_graph, graph, title, links, lock) for title, links in links_dict.items()]
        for future in as_completed(futures):
            future.result()    
    return graph


def build_graph_and_categories(start_page, depth, min_edges=35, min_nodes_in_cat=15, verbosity=2):
    # Build the initial graph
    graph = build_wikipedia_graph(start_page, depth, verbosity)

    # Finish the graph by fetching links and adding missing edges
    if verbosity >= 1:
        print('-'*50, flush=True)
        print('Finishing graph by adding edges between already existing nodes...', flush=True)
    graph = finish_graph(graph)
    if verbosity >= 1:
        print('')
        print('Graph finished', flush=True)
        print('Number of nodes:', graph.number_of_nodes(), 'Number of edges:', graph.number_of_edges(), flush=True)
        print('')

        if verbosity >= 2:
            # Print the 10 nodes with the highest degree
            print('Top 10 nodes by degree:')
            sorted_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
            for node, degree in sorted_nodes[:10]:
                print(f'{node}: {degree} edges')
            print('')

    if verbosity >= 1:
        print('-'*50, flush=True)
        print(f'Removing nodes with degree <= {min_edges}...', flush=True)
    # remove nodes with degree <= 2
    nodes_to_remove = [node for node, degree in dict(graph.degree).items() if degree <= min_edges]
    graph.remove_nodes_from(nodes_to_remove)
    if verbosity >= 1:
        print('Final number of nodes:', graph.number_of_nodes(), 'Number of edges:', graph.number_of_edges(), flush=True)
        print('')


    # Categorize the graph
    if verbosity >= 1:
        print('-'*50, flush=True)
        print('Building categories...', flush=True)
    category_dict = categorize_graph(graph)

    # remove categories with less than 10 nodes
    categories_to_remove = [category for category, nodes in category_dict.items() if len(nodes) <= min_nodes_in_cat]
    for category in categories_to_remove:
        del category_dict[category]

    if verbosity >= 1:
        print('Categories built', flush=True)
        print('Number of categories:', len(category_dict), flush=True)
        print('')
        if verbosity >= 2:
            # print the 10 categories with the most pages and the number of pages in each
            print('Top 10 categories by number of pages:')
            sorted_categories = sorted(category_dict.items(), key=lambda x: len(x[1]), reverse=True)
            for category, pages in sorted_categories[:10]:
                print(f'{category}: {len(pages)} pages')
            print('')
    
    return graph, category_dict
