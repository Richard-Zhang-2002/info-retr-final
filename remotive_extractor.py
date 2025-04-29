import requests
import re
import os
import json
import csv
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class RemotiveJobExtractor:
    """Full extractor that fetches jobs, extracts metadata, matches resume, and can save results."""

    def __init__(self):
        self.api_url = "https://remotive.com/api/remote-jobs"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_jobs(self, category: str = None, company_name: str = None, search: str = None, limit: int = 50) -> Dict[str, Any]:
        params = {}
        if category:
            params["category"] = category
        if company_name:
            params["company_name"] = company_name
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
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_salary_range(self, salary_text: str, description: str) -> Optional[Tuple[float, float]]:
        if salary_text and isinstance(salary_text, str):
            salary_patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]\s*[-–to]+\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]'
            ]
            for pattern in salary_patterns:
                matches = re.search(pattern, salary_text, re.IGNORECASE)
                if matches:
                    try:
                        min_salary = float(matches.group(1).replace(',', ''))
                        max_salary = float(matches.group(2).replace(',', ''))
                        if 'k' in salary_text.lower():
                            min_salary *= 1000
                            max_salary *= 1000
                        return (min_salary, max_salary)
                    except (ValueError, IndexError):
                        continue

        if description:
            clean_description = re.sub(r'<[^>]+>', ' ', description)
            patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
            ]
            for pattern in patterns:
                matches = re.search(pattern, clean_description, re.IGNORECASE)
                if matches:
                    try:
                        min_salary = float(matches.group(1).replace(',', ''))
                        max_salary = float(matches.group(2).replace(',', ''))
                        return (min_salary, max_salary)
                    except (ValueError, IndexError):
                        continue
        return None

    def extract_experience(self, description: str) -> Optional[int]:
        if not description:
            return None
        clean_description = re.sub(r'<[^>]+>', ' ', description)
        patterns = [
            r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience|exp)'
        ]
        for pattern in patterns:
            matches = re.search(pattern, clean_description, re.IGNORECASE)
            if matches:
                try:
                    return int(matches.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def extract_location(self, required_location: str, description: str) -> Optional[str]:
        if required_location and isinstance(required_location, str) and required_location.strip() != "":
            return required_location.strip()
        if not description:
            return None

        clean_description = re.sub(r'<[^>]+>', ' ', description)
        patterns = [
            r'location:\s*(.*?)(?:\.|,|\n)'
        ]
        for pattern in patterns:
            matches = re.search(pattern, clean_description, re.IGNORECASE)
            if matches:
                return matches.group(1).strip()

        work_arrangements = [
            (r'\bremote\b', "Remote"),
            (r'\bhybrid\b', "Hybrid"),
            (r'\bon[-\s]site\b', "On-site"),
            (r'\bin[-\s]office\b', "In-office")
        ]
        for pattern, arrangement in work_arrangements:
            if re.search(pattern, clean_description, re.IGNORECASE):
                return arrangement
        return None

    def process_job_descriptions(self, category: str = None, company_name: str = None, search: str = None, limit: int = 50) -> List[Dict]:
        response = self.fetch_jobs(category, company_name, search, limit)
        jobs = response.get("jobs", [])

        results = []
        for job in jobs:
            description = job.get("description", "")
            salary_text = job.get("salary", "")
            required_location = job.get("candidate_required_location", "")

            salary_range = self.extract_salary_range(salary_text, description)
            experience = self.extract_experience(description)
            location = self.extract_location(required_location, description)

            min_salary, max_salary = (None, None)
            if salary_range:
                min_salary, max_salary = salary_range

            job_info = {
                "job_id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company_name"),
                "url": job.get("url"),
                "category": job.get("category"),
                "job_type": job.get("job_type"),
                "salary_field": salary_text,
                "min_salary": min_salary,
                "max_salary": max_salary,
                "years_experience": experience,
                "location_field": required_location,
                "location": location,
                "publication_date": job.get("publication_date"),
                "description": self.clean_text(description)
            }
            results.append(job_info)

        return results

    def load_resume(self, filepath: str) -> str:
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return text.strip()
        else:
            raise ValueError("Unsupported file type.")

    def match_resume_to_jobs(self, resume_text: str, jobs: List[Dict], top_n: int = 5) -> List[Dict]:
        corpus = [resume_text] + [job["description"] for job in jobs]
        embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True)

        resume_embedding = embeddings[0].cpu().numpy()
        job_embeddings = embeddings[1:].cpu().numpy()

        cosine_sim = (resume_embedding @ job_embeddings.T) / (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embeddings, axis=1))

        for idx, job in enumerate(jobs):
            job["similarity_score"] = cosine_sim[idx]

        sorted_jobs = sorted(jobs, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_jobs[:top_n]

    def save_results(self, results: List[Dict], format: str = "json", filename: str = None) -> str:
        format = format.lower()
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"remotive_job_data_{timestamp}.{format}"

        if format == "json":
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif format == "csv":
            fieldnames = set()
            for job in results:
                fieldnames.update(job.keys())
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(results)
        elif format == "excel":
            df = pd.DataFrame(results)
            df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported file format: must be 'json', 'csv', or 'excel'.")

        return os.path.abspath(filename)
