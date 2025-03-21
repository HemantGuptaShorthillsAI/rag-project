import unittest
from unittest.mock import patch, MagicMock
from scrap import UnicornScraper  # Assuming your scrap script is named 'scrap.py'
import os

class TestUnicornScraper(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("1")
        cls.test_url = "https://en.wikipedia.org/wiki/List_of_unicorn_startup_companies"
        cls.scrap = UnicornScraper(cls.test_url)
    
    def test_clean_text(self):
        print("2")
        text = "  Hello  World  [1] [2]  "
        cleaned_text = self.scrap.clean_text(text)
        self.assertEqual(cleaned_text, "Hello World")
    
    @patch("scrap.requests.get")
    def test_get_unicorn_startups_success(self, mock_get):
        print("3")
        mock_response = MagicMock()
        with open("test_data/unicorn_list.html", "r", encoding="utf-8") as f:
            mock_response.text = f.read()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        unicorns = self.scrap.get_unicorn_startups()
        self.assertGreater(len(unicorns), 0)
        self.assertIn("name", unicorns[0])
        self.assertIn("industry", unicorns[0])
    
    @patch("scrap.requests.get")
    def test_get_unicorn_startups_failure(self, mock_get):
        print("4")
        mock_get.return_value.status_code = 404
        unicorns = self.scrap.get_unicorn_startups()
        self.assertEqual(unicorns, [])

    @patch("scrap.requests.get")
    def test_scrape_startup_page_success(self, mock_get):
        print("5")
        mock_response = MagicMock()
        with open("test_data/startup_page.html", "r", encoding="utf-8") as f:
            mock_response.text = f.read()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        startup = {"name": "Stripe", "wiki_url": "https://en.wikipedia.org/wiki/Stripe"}
        self.scrap.scrape_startup_page(startup, 1)

        expected_filename = os.path.join(self.scrap.output_folder, "1_Stripe.txt")
        self.assertTrue(os.path.exists(expected_filename))

    @patch("scrap.requests.get")
    def test_scrape_startup_page_failure(self, mock_get):
        print("6")
        mock_get.return_value.status_code = 404
        startup = {"name": "Binance", "wiki_url": "https://en.wikipedia.org/wiki/Binance"}
        self.scrap.scrape_startup_page(startup, 2)
        
        expected_filename = os.path.join(self.scrap.output_folder, "2_Binance.txt")
        self.assertFalse(os.path.exists(expected_filename))
    
    @patch("scrap.requests.get")
    def test_scrape_unicorns(self, mock_get):
        print("7")
        """ Fixing the mock issue to ensure get_unicorn_startups() returns test data """
        mock_response = MagicMock()
        with open("test_data/unicorn_list.html", "r", encoding="utf-8") as f:
            mock_response.text = f.read()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        mock_startups = [
            {"name": "Stripe", "wiki_url": "https://en.wikipedia.org/wiki/Stripe"},
            {"name": "Binance", "wiki_url": "https://en.wikipedia.org/wiki/Binance"}
        ]
        
        with patch.object(self.scrap, "get_unicorn_startups", return_value=mock_startups):
            with patch.object(self.scrap, "scrape_startup_page", return_value=None) as mock_scrape:
                self.scrap.scrape_unicorns()
                self.assertGreater(mock_scrape.call_count, 0, "scrape_startup_page was not called")

if __name__ == "__main__":
    unittest.main()
