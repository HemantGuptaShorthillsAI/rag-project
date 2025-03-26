import requests
from bs4 import BeautifulSoup
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class UnicornScraper:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self, url):
        self.URL = url
        self.output_folder = "unicorn_startups"
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        return text.replace("\n", " ").strip()

    def get_unicorn_startups(self):
        response = requests.get(self.URL, headers=self.HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch Wikipedia page. Status Code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", {"class": "wikitable"})
        if not tables:
            print("No tables found on Wikipedia page!")
            return []

        unicorn_table = None
        for table in tables:
            headers = table.find_all("th")
            if any("Company" in th.text for th in headers):
                unicorn_table = table
                break

        if not unicorn_table:
            print("Unicorn startups table not found!")
            return []

        unicorns = []
        rows = unicorn_table.find_all("tr")[1:]

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            company_name = cols[0].text.strip()
            industry = cols[1].text.strip()
            country = cols[2].text.strip()
            valuation = cols[3].text.strip()
            founded_year = cols[4].text.strip()
            company_link = cols[0].find("a")

            startup_info = {
                "name": company_name,
                "industry": industry,
                "country": country,
                "valuation": valuation,
                "founded": founded_year,
                "wiki_url": "https://en.wikipedia.org" + company_link["href"] if company_link else None
            }
            unicorns.append(startup_info)
        
        return unicorns

    def scrape_startup_page(self, startup, index):
        if not startup["wiki_url"]:
            return
        
        retries = 2  # Reduce retries to save time
        while retries > 0:
            try:
                response = requests.get(startup["wiki_url"], headers=self.HEADERS, timeout=7)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                retries -= 1
                time.sleep(2)  # Reduce sleep time

        if retries == 0:
            print(f"Skipping {startup['name']} due to repeated failures.")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        full_text = self.clean_text(content_div.get_text(separator=" ")) if content_div else "No content found"

        filename = f"{self.output_folder}/{index}_{startup['name'].replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Startup Name: {startup['name']}\n")
            f.write(f"Wiki URL: {startup['wiki_url']}\n\n")
            f.write(full_text)

    def scrape_unicorns(self):
        startups = self.get_unicorn_startups()
        scraped_files = {f.split("_")[0] for f in os.listdir(self.output_folder) if f.endswith(".txt")}

        with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust threads based on system capability
            future_to_startup = {
                executor.submit(self.scrape_startup_page, startup, index): startup
                for index, startup in enumerate(startups, start=1)
                if str(index) not in scraped_files
            }

            for future in as_completed(future_to_startup):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error scraping {future_to_startup[future]['name']}: {e}")

        print("Scraping completed!")

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_unicorn_startup_companies"
    scraper = UnicornScraper(url)
    scraper.scrape_unicorns()
