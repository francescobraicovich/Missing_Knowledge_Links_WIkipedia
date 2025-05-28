import unittest
import numpy as np
from src.missing_links_analyzer.utils import filter_sentences

class TestFilterSentences(unittest.TestCase):

    def test_empty_input_array(self):
        self.assertEqual(filter_sentences(np.array([]), ["bad"]), set())

    def test_empty_filter_list(self):
        self.assertEqual(filter_sentences(np.array(["good sentence", "another one"]), []), 
                         {"good sentence", "another one"})

    def test_no_matching_words(self):
        self.assertEqual(filter_sentences(np.array(["clean", "perfectly fine"]), ["bad", "ugly"]), 
                         {"clean", "perfectly fine"})

    def test_some_matching_words(self):
        self.assertEqual(filter_sentences(np.array(["this is bad", "this is good", "really bad stuff"]), ["bad"]), 
                         {"this is good"})

    def test_all_matching_words(self):
        self.assertEqual(filter_sentences(np.array(["bad sentence", "very bad indeed"]), ["bad"]), 
                         set())

    def test_case_insensitivity_in_sentence(self):
        # Filter word is lowercase, sentence has mixed case
        self.assertEqual(filter_sentences(np.array(["Sentence with BadWord", "good one"]), ["badword"]), 
                         {"good one"})

    def test_case_insensitivity_in_filter_word(self):
        # Filter word is mixed case, sentence has lowercase
        self.assertEqual(filter_sentences(np.array(["sentence with badword", "good one"]), ["BaDwOrD"]), 
                         {"good one"})

    def test_non_numpy_input_list_of_strings(self):
        self.assertEqual(filter_sentences(["list sentence bad", "good list entry"], ["bad"]), 
                         {"good list entry"})

    def test_non_numpy_input_tuple_of_strings(self):
        self.assertEqual(filter_sentences(("tuple sentence bad", "good tuple entry"), ["bad"]), 
                         {"good tuple entry"})

    def test_mixed_types_in_input_list(self):
        # utils.py converts inputs to string. "123" and "None" don't contain "bad".
        self.assertEqual(filter_sentences(["list sentence bad", "good list entry", 123, None], ["bad"]), 
                         {"good list entry", "123", "None"})
    
    def test_filter_words_with_special_chars(self):
        # Assuming special chars are treated literally in substring search
        self.assertEqual(filter_sentences(np.array(["sentence with $pecial", "normal"]), ["$pecial"]),
                         {"normal"})

    def test_sentences_with_leading_trailing_spaces(self):
        # filter_sentences should handle this as np.char.lower and np.char.find operate on elements
        self.assertEqual(filter_sentences(np.array(["  bad sentence  ", "good one"]), ["bad"]),
                         {"good one"})
        # The original sentence (with spaces) should be returned if it doesn't match
        self.assertEqual(filter_sentences(np.array(["  another bad one  ", "  perfectly fine  "]), ["bad"]),
                         {"  perfectly fine  "})

    def test_empty_strings_in_input_array(self):
        # Empty strings do not contain "bad"
        self.assertEqual(filter_sentences(np.array(["", "good", "also bad", ""]), ["bad"]),
                         {"", "good"})
                         
    def test_filter_list_contains_non_string(self):
        # words_to_filter elements are converted to string and lowercased
        self.assertEqual(filter_sentences(np.array(["sentence with 123", "good one"]), [123, "word"]),
                         {"good one"}) # "sentence with 123" filtered by 123 -> "123"
        self.assertEqual(filter_sentences(np.array(["sentence with None", "good one"]), [None, "word"]),
                         {"good one"}) # "sentence with None" filtered by None -> "none"


if __name__ == '__main__':
    unittest.main()
```
