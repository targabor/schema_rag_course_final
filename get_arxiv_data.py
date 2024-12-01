import sys
from paperscraper.arxiv import get_arxiv_papers
from paperscraper.pdf import save_pdf
import os


def get_papers(keywords, max_results=10):
    # Define the search query
    query = " OR ".join([f"\"{keyword}\"" for keyword in keywords])
    print(f"Query: {query}\nWith {max_results} results")
    # Get the papers
    papers = get_arxiv_papers(query, max_results=max_results)
    
    # Create data folder if not exists
    if not os.path.exists("data"):
        os.makedirs("data")
    # Download PDFs
    papers.apply(lambda x: save_pdf(dict(x), f"data/{x['doi'].replace('/', '_')}.pdf"), axis=1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_arxiv_data.py <keyword1> <keyword2> ... [max_results]")
        sys.exit(1)
    
    keywords = sys.argv[1:-1]
    max_results = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 10
    
    if not sys.argv[-1].isdigit():
        keywords.append(sys.argv[-1])
    
    get_papers(keywords, max_results)