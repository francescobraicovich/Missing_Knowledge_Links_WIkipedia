import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from threading import Lock

blacklist_substrings = {'wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes'}

def get_wikipedia_categories(session, title):
    """Fetches the categories for a given Wikipedia page title using the Wikipedia API."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'categories',
        'titles': title
    }
    response = session.get(url, params=params)
    data = response.json()
    pages = data['query']['pages']
    categories = []

    for page_id in pages:
        if 'categories' in pages[page_id]:
            categories.extend([cat['title'] for cat in pages[page_id]['categories']])

    return categories

def add_categories_to_dict(category_dict, lock, title, categories):
    """Adds categories to the dictionary in a thread-safe manner."""
    with lock:
        for category in categories:
            cleaned_category = category.replace('Category:', '')

            if not any(substring in cleaned_category.lower() for substring in blacklist_substrings):
                if cleaned_category not in category_dict:
                    category_dict[cleaned_category] = []
                category_dict[cleaned_category].append(title)

def categorize_graph(graph):
    """Categorizes Wikipedia pages from a given nx graph."""
    # Extract the page titles from the graph nodes
    page_titles = list(graph.nodes)
    
    # Use a dictionary to hold categories and their corresponding pages
    category_dict = {}

    # Use a session for connection pooling
    session = requests.Session()
    
    # Use a lock for thread-safe updates to the category dictionary
    lock = Lock()

    # Fetch categories in parallel
    with ThreadPoolExecutor(max_workers=None) as executor:
        future_to_title = {executor.submit(get_wikipedia_categories, session, title): title for title in page_titles}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                categories = future.result()
                add_categories_to_dict(category_dict, lock, title, categories)
            except Exception as exc:
                print(f'Error fetching categories for {title}: {exc}')
    
    return category_dict
