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


class MuseJobExtractor:
    """Full extractor that fetches jobs, extracts metadata, matches resume, and can save results."""

    def __init__(self, api_key=None):
        self.api_url = "https://www.themuse.com/api/public/jobs"
        self.api_key = api_key  # Optional API key for higher rate limits
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_jobs(self, category: str = None, company_name: str = None, 
                  search: str = None, location: str = None, level: str = None,
                  page: int = 1, descending: bool = False) -> Dict[str, Any]:
        """
        Fetch jobs from The Muse API.
        
        Args:
            category: Job category
            company_name: Company name filter
            search: Search term
            location: Job location filter
            level: Experience level filter
            page: Page number (1-indexed)
            descending: Whether to show descending results
            
        Returns:
            Dictionary containing job data
        """
        params = {'page': page}
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        if category:
            params['category'] = category
        if company_name:
            params['company'] = company_name
        if location:
            params['location'] = location
        if level:
            params['level'] = level
        if descending:
            params['descending'] = 'true'

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            
            # Print rate limit information for debugging
            limit = response.headers.get('X-RateLimit-Limit')
            remaining = response.headers.get('X-RateLimit-Remaining')
            reset = response.headers.get('X-RateLimit-Reset')
            print(f"Rate limits - Total: {limit}, Remaining: {remaining}, Reset in: {reset}s")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs: {e}")
            return {"results": [], "page_count": 0, "page": 0}

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_salary_range(self, contents: str) -> Optional[Tuple[float, float]]:
        """Extract salary range from job description."""
        if not contents:
            return None
            
        clean_contents = re.sub(r'<[^>]+>', ' ', contents)
        
        # Common salary patterns
        patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]\s*[-–to]+\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]',
            r'salary.*?\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'pay.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]\s*[-–to]+\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]',
            r'compensation.*?\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, clean_contents, re.IGNORECASE)
            if matches:
                try:
                    min_salary = float(matches.group(1).replace(',', ''))
                    max_salary = float(matches.group(2).replace(',', ''))
                    if 'k' in clean_contents[matches.start():matches.end()].lower():
                        min_salary *= 1000
                        max_salary *= 1000
                    return (min_salary, max_salary)
                except (ValueError, IndexError):
                    continue
        return None

    def extract_experience(self, contents: str) -> Optional[int]:
        """Extract years of experience required from job description."""
        if not contents:
            return None
            
        clean_contents = re.sub(r'<[^>]+>', ' ', contents)
        
        patterns = [
            r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience|exp)',
            r'minimum\s*(?:of)?\s*(\d+)\s*(?:year|yr)',
            r'at least\s*(\d+)\s*(?:year|yr)'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, clean_contents, re.IGNORECASE)
            if matches:
                try:
                    return int(matches.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def extract_location_type(self, locations: List[Dict], contents: str) -> Optional[str]:
        """Determine if job is remote, hybrid, or on-site based on location and description."""
        if not contents:
            return None
            
        clean_contents = re.sub(r'<[^>]+>', ' ', contents).lower()
        
        # Check location names first
        location_names = [loc.get('name', '').lower() for loc in locations if loc.get('name')]
        
        for loc in location_names:
            if 'remote' in loc:
                return "Remote"
            if 'hybrid' in loc:
                return "Hybrid"
        
        # Then check description
        work_arrangements = [
            (r'\bremote\b', "Remote"),
            (r'\bhybrid\b', "Hybrid"),
            (r'\bon[-\s]site\b', "On-site"),
            (r'\bin[-\s]office\b', "In-office")
        ]
        
        for pattern, arrangement in work_arrangements:
            if re.search(pattern, clean_contents, re.IGNORECASE):
                return arrangement
                
        # Default to location name if we can't determine
        if locations and len(locations) > 0:
            return locations[0].get('name')
            
        return None

    def process_job_descriptions(self, category: str = None, company_name: str = None, 
                                search: str = None, location: str = None, level: str = None,
                                page: int = 1, limit: int = 20) -> List[Dict]:
        """
        Process job descriptions from The Muse API.
        
        Returns:
            List of dictionaries containing processed job information
        """
        results = []
        current_page = page
        seen_jobs = set() 
    
        while len(results) < limit:
            response = self.fetch_jobs(category, company_name, search, location, level, page=current_page)
            jobs = response.get("results", [])
            if not jobs:
                break
            for job in jobs:  
                # Limit to specified number of results
                if len(results) >= limit:
                    break
                contents = job.get("contents", "")
                key = (job.get("name", "").strip().lower(), self.clean_text(contents).strip().lower())
                if key in seen_jobs:
                    continue
                seen_jobs.add(key)
                # Get locations and categories
                locations = job.get("locations", [])
                categories = job.get("categories", [])
                levels = job.get("levels", [])
                
                # Extract additional information
                salary_range = self.extract_salary_range(contents)
                experience = self.extract_experience(contents)
                location_type = self.extract_location_type(locations, contents)
                
                min_salary, max_salary = (None, None)
                if salary_range:
                    min_salary, max_salary = salary_range
                
                # Prepare location and category strings
                location_names = [loc.get("name", "") for loc in locations]
                category_names = [cat.get("name", "") for cat in categories]
                level_names = [level.get("name", "") for level in levels]
                
                job_info = {
                    "job_id": job.get("id"),
                    "title": job.get("name"),
                    "company": job.get("company", {}).get("name") if job.get("company") else None,
                    "url": job.get("refs", {}).get("landing_page") if job.get("refs") else None,
                    "category": ", ".join(category_names) if category_names else None,
                    "job_type": job.get("type"),
                    "min_salary": min_salary,
                    "max_salary": max_salary,
                    "years_experience": experience,
                    "locations": ", ".join(location_names) if location_names else None,
                    "location_type": location_type,
                    "levels": ", ".join(level_names) if level_names else None,
                    "publication_date": job.get("publication_date"),
                    "description": self.clean_text(contents)
                }
                results.append(job_info)
            current_page += 1
        
        print("number of job found:",len(results))
        return results

    def load_resume(self, filepath: str) -> str:
        """Load resume text from file."""
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
        """Match resume to jobs using TF-IDF and cosine similarity."""
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
        """Save results to file in specified format."""
        format = format.lower()
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"muse_job_data_{timestamp}.{format}"

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