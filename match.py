import requests
import re
import os
import pdfplumber
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RemotiveJobExtractor:
    """Class to fetch remote job descriptions from Remotive API, extract key information, and match to a resume."""
    
    def __init__(self):
        self.api_url = "https://remotive.com/api/remote-jobs"
    
    def fetch_jobs(self, category: str = None, search: str = None, limit: int = 50) -> Dict[str, Any]:
        params = {}
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        if limit:
            params["limit"] = limit
        
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs: {e}")
            return {"job-count": 0, "jobs": []}
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_jobs_for_matching(self, category: str = None, search: str = None, limit: int = 50) -> List[Dict]:
        """Fetch and clean jobs."""
        response = self.fetch_jobs(category=category, search=search, limit=limit)
        jobs = response.get("jobs", [])
        
        job_list = []
        for job in jobs:
            job_entry = {
                "job_id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company_name"),
                "description": self.clean_text(job.get("description", "")),
                "location": job.get("candidate_required_location", "N/A"),  # Fetch location info
                "salary": job.get("salary", "N/A"),  # Fetch salary info
                "experience": job.get("experience_required", "N/A"),  # Fetch experience info
                "url": job.get("url"),
            }
            job_list.append(job_entry)
        
        return job_list

    def load_resume(self, filepath: str) -> str:
        """Load resume from a .txt or .pdf file."""
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + " "
            return text.strip()
        else:
            raise ValueError("Unsupported file type. Please provide a .txt or .pdf resume.")

    def match_resume_to_jobs(self, resume_text: str, jobs: List[Dict], top_n: int = 5) -> List[Dict]:
        """Match a resume text to jobs using TF-IDF and cosine similarity."""
        corpus = [resume_text] + [job["description"] for job in jobs]

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        for idx, job in enumerate(jobs):
            job["similarity_score"] = cosine_sim[idx]
        
        sorted_jobs = sorted(jobs, key=lambda x: x["similarity_score"], reverse=True)
        
        return sorted_jobs[:top_n]

# Example usage
if __name__ == "__main__":
    extractor = RemotiveJobExtractor()

    # Choose your resume file (either .txt or .pdf)
    resume_path = "Zayn_Zaidi_Resume.pdf"  # or "resume.txt"

    # Load the resume
    resume_text = extractor.load_resume(resume_path)

    # Fetch jobs
    jobs = extractor.process_jobs_for_matching(category="Software Development", limit=50)

    # Match resume
    top_matches = extractor.match_resume_to_jobs(resume_text, jobs, top_n=5)

    # Display results
    for match in top_matches:
        print(f"Title: {match['title']}")
        print(f"Company: {match['company']}")
        print(f"Location: {match['location']}")
        print(f"Salary: {match['salary']}")
        print(f"Experience Required: {match['experience']}")
        print(f"URL: {match['url']}")
        print(f"Similarity Score: {match['similarity_score']:.4f}")
        print("="*60)