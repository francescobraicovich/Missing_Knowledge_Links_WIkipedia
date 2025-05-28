"""
General utility functions for the Missing Links Analyzer.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_sentences(sentences: np.ndarray, words_to_filter: list[str]) -> set[str]:
    """
    Filters out sentences that contain any of the specified words.

    Args:
        sentences (np.ndarray): An array of sentences to filter.
        words_to_filter (list[str]): A list of words to filter out. 
                               Sentences containing these words (case-insensitive) will be removed.

    Returns:
        set[str]: A set of filtered sentences (those not containing any blacklisted words).
    """
    if not isinstance(sentences, np.ndarray):
        logger.warning("Input 'sentences' is not a numpy array. Attempting to convert.")
        try:
            # Attempt to convert to a numpy array of strings
            sentences = np.array(sentences, dtype=str) 
        except Exception as e:
            logger.error(f"Could not convert 'sentences' to a numpy array of strings: {e}")
            return set() # Return empty set on failure
    
    # Handle cases where sentences might be empty or effectively scalar after conversion
    if sentences.ndim == 0 or sentences.size == 0:
        logger.debug("Input 'sentences' is empty or scalar. Returning empty set.")
        return set()

    # Ensure all elements are strings for np.char operations, then lowercase
    # This also handles cases where sentences might be like np.array([None]) or np.array([1, "foo"])
    try:
        sentences_str = sentences.astype(str) 
    except Exception as e: # Should be rare if initial conversion to str array worked
        logger.error(f"Could not ensure all elements in 'sentences' are strings: {e}")
        return set()
        
    sentences_lower = np.char.lower(sentences_str)
    
    # Initialize mask to False (all sentences kept initially)
    combined_mask = np.zeros(len(sentences_lower), dtype=bool)

    for word in words_to_filter:
        if not isinstance(word, str): # Ensure word is a string
            logger.warning(f"Non-string element '{word}' found in words_to_filter. Converting to string.")
            word = str(word)
        
        word_lower = word.lower()
        
        # Create a mask for the current word: True if word is found in a sentence
        # np.char.find returns -1 if not found, >= 0 if found.
        current_word_mask = np.char.find(sentences_lower, word_lower) != -1
        
        # Combine masks: if a sentence matches any word, it should be masked (True)
        combined_mask |= current_word_mask
    
    # Apply the combined_mask to the original sentences array (sentences_str, not sentences_lower)
    # We want to keep sentences where the mask is False (i.e., no forbidden words found)
    filtered_sentences_array = sentences_str[~combined_mask]
    
    return set(filtered_sentences_array)

if __name__ == '__main__':
    # Example Usage for testing
    logger.setLevel(logging.INFO) # Configure logger for example
    logging.basicConfig() # Basic console logging

    test_sentences_np = np.array([
        "This is a normal sentence.",
        "This sentence contains a WIKI link.",
        "Another example with a TEMPLATE.",
        "No forbidden words here.",
        "Short one.",
        "sentence with cs1 reference"
    ], dtype=object) # Using dtype=object to test mixed types if any

    blacklist = ["wiki", "template", "cs1"]

    print(f"Original Sentences (type: {type(test_sentences_np)}): \n{test_sentences_np}\n")
    filtered_set = filter_sentences(test_sentences_np, blacklist)
    print(f"Filtered Sentences (type: {type(filtered_set)}): \n{filtered_set}\n")

    assert "This is a normal sentence." in filtered_set
    assert "No forbidden words here." in filtered_set
    assert "Short one." in filtered_set
    assert "This sentence contains a WIKI link." not in filtered_set
    assert "Another example with a TEMPLATE." not in filtered_set
    assert "sentence with cs1 reference" not in filtered_set
    print("Assertions passed for numpy array input.")

    # Test with list input
    test_sentences_list = [
        "List sentence normal.",
        "List sentence with wiki.",
        "List sentence with template.",
        123, # Test non-string in list
        None # Test None in list
    ]
    print(f"Original Sentences (type: {type(test_sentences_list)}): \n{test_sentences_list}\n")
    filtered_set_list = filter_sentences(test_sentences_list, blacklist)
    print(f"Filtered Sentences from list (type: {type(filtered_set_list)}): \n{filtered_set_list}\n")
    assert "List sentence normal." in filtered_set_list
    assert "List sentence with wiki." not in filtered_set_list
    assert "123" not in filtered_set_list # Should be filtered if it contains "1" and "1" is in blacklist (not here) or kept if not.
                                        # In this case, "123" doesn't contain "wiki", "template", or "cs1".
                                        # The provided code handles this correctly.
    assert "None" not in filtered_set_list # `None` becomes "None" string, then filtered if "none" is in blacklist.
    
    # After reviewing the code, 123 and None will be converted to "123" and "None"
    # If "123" or "None" (lowercase) don't contain items from blacklist, they should be in filtered set.
    if "123" not in blacklist and not any(b in "123" for b in blacklist):
        assert "123" in filtered_set_list
    if "none" not in blacklist and not any(b in "none" for b in blacklist) : # 'None' becomes 'none'
        assert "None" in filtered_set_list # The original value 'None' (as a string) is returned
    
    print("Assertions passed for list input.")

    # Test with empty sentences
    empty_np_array = np.array([])
    filtered_empty_np = filter_sentences(empty_np_array, blacklist)
    assert len(filtered_empty_np) == 0
    print(f"Filtered empty numpy array: {filtered_empty_np}")

    empty_list = []
    filtered_empty_list = filter_sentences(empty_list, blacklist)
    assert len(filtered_empty_list) == 0
    print(f"Filtered empty list: {filtered_empty_list}")
    print("Assertions for empty inputs passed.")

    # Test with empty blacklist
    filtered_empty_blacklist = filter_sentences(test_sentences_np, [])
    assert len(filtered_empty_blacklist) == len(set(test_sentences_np.astype(str))) # All should be kept
    print(f"Filtered with empty blacklist: {filtered_empty_blacklist}")
    print("Assertion for empty blacklist passed.")

    # Test with words_to_filter containing non-string
    non_string_blacklist = ["wiki", 123, "template"]
    filtered_non_string_blacklist = filter_sentences(test_sentences_np, non_string_blacklist)
    # Expected: 123 becomes "123". If "123" is in a sentence, it's filtered.
    # "This sentence contains a WIKI link." -> filtered
    # "Another example with a TEMPLATE." -> filtered
    # "sentence with cs1 reference" -> kept, as "123" is not "cs1"
    assert "sentence with cs1 reference" in filtered_non_string_blacklist
    print(f"Filtered with non-string in blacklist: {filtered_non_string_blacklist}")
    print("Assertion for non-string in blacklist passed.")

    logger.info("All example tests for filter_sentences completed.")
