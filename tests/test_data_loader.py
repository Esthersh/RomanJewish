import unittest
import os
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.keywords_path = "Keywords_05022026.csv"
        self.corpus_path = "LUR sample corpus.xlsx"

    def test_load_keywords(self):
        if not os.path.exists(self.keywords_path):
            self.skipTest("Keywords file not found")
        
        keywords = self.loader.load_keywords(self.keywords_path)
        self.assertTrue(len(keywords) > 0)
        first_kw = keywords[0]
        self.assertIsNotNone(first_kw.id)
        self.assertIsNotNone(first_kw.name)

    def test_load_corpus(self):
        if not os.path.exists(self.corpus_path):
            self.skipTest("Corpus file not found")
            
        samples = self.loader.load_corpus(self.corpus_path)
        self.assertTrue(len(samples) > 0)
        # We can add more specific assertions after we know exact column names

if __name__ == '__main__':
    unittest.main()
