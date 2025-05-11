import requests
import re
import os
import json
import csv
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from groq import Groq
import geopy.distance
from geopy.geocoders import Nominatim
from collections import defaultdict

class MuseJobExtractor:
    """Full extractor that fetches jobs, extracts metadata, matches resume, and can save results with fuzzy range matching for location, salary, and experience."""

    def __init__(self, api_key=None):
        
        self.api_url = "https://www.themuse.com/api/public/jobs"
        self.api_key = api_key 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize geocoder for location range matching
        self.geocoder = Nominatim(user_agent="muse_job_extractor")
        self.location_cache = {}  # Cache for geocoding results

        try:
            with open("openai_key.txt", "r") as f:
                key = f.read().strip()
                self.client = Groq(api_key=key)
        except FileNotFoundError:
            print("Warning: openai_key.txt not found. Resume summarization will fail.")

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
        if search:
            params['search'] = search

        
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
    
    def preprocess_job_description(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return ' '.join(text.lower().split())
    
    def summarize_resume(self, resume_text: str) -> str:
        response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "user", "content": f"Summarize this resume into a few coherent paragraphs:\n{resume_text}"}
                    ],
                    temperature=0.3
                )
        return response.choices[0].message.content.strip()

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

    def geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get latitude and longitude coordinates for a location name with caching."""
        if not location_name:
            return None
            
        # Check cache first
        if location_name in self.location_cache:
            return self.location_cache[location_name]
            
        try:
            # Try to geocode the location
            location = self.geocoder.geocode(location_name)
            if location:
                coordinates = (location.latitude, location.longitude)
                self.location_cache[location_name] = coordinates
                return coordinates
        except Exception as e:
            print(f"Error geocoding {location_name}: {e}")
            
        return None
        
    def calculate_location_distance(self, location1: str, location2: str) -> Optional[float]:
        """Calculate the distance between two locations in miles."""
        coords1 = self.geocode_location(location1)
        coords2 = self.geocode_location(location2)
        
        if coords1 and coords2:
            return geopy.distance.distance(coords1, coords2).miles
        return None
        
    def is_within_location_range(self, job_location: str, desired_location: str, max_distance_km: float) -> bool:
        """Check if job location is within the specified distance of the desired location."""
        if not job_location or not desired_location:
            return False
            
        # Remote jobs match any location preference
        if "remote" in job_location.lower():
            return True
            
        # Check exact match first
        if desired_location.lower() in job_location.lower():
            return True
            
        # Calculate distance
        distance = self.calculate_location_distance(job_location, desired_location)
        return distance is not None and distance <= max_distance_km
    
    def is_within_salary_range(self, job_min: Optional[float], job_max: Optional[float], 
                              desired_min: Optional[float], desired_max: Optional[float], 
                              tolerance_percent: float = 15.0) -> bool:
        """
        Check if job salary range has overlap with desired range, with tolerance.
        
        Args:
            job_min: Minimum job salary
            job_max: Maximum job salary
            desired_min: Minimum desired salary
            desired_max: Maximum desired salary
            tolerance_percent: Percentage tolerance for matching (e.g., 15% below desired min still matches)
            
        Returns:
            Boolean indicating if ranges overlap
        """
        if job_min is None and job_max is None:
            # If job has no salary info, always include it in results
            return True
            
        if desired_min is None and desired_max is None:
            # If no desired salary specified, include all jobs
            return True
            
        # Apply tolerance to desired minimum
        if desired_min is not None:
            min_with_tolerance = desired_min * (1 - tolerance_percent / 100)
        else:
            min_with_tolerance = float('-inf')
            
        # Apply tolerance to desired maximum
        if desired_max is not None:
            max_with_tolerance = desired_max * (1 + tolerance_percent / 100)
        else:
            max_with_tolerance = float('inf')
            
        # Check overlap scenarios
        if job_min is not None and job_max is not None:
            # Case 1: Job has both min and max salary
            return not (job_max < min_with_tolerance or job_min > max_with_tolerance)
        elif job_min is not None:
            # Case 2: Job has only min salary
            return job_min <= max_with_tolerance
        elif job_max is not None:
            # Case 3: Job has only max salary
            return job_max >= min_with_tolerance
            
        return False
        
    def is_within_experience_range(self, job_years: Optional[int], desired_years: Optional[int], 
                                 tolerance_years: int = 2) -> bool:
        """
        Check if job experience requirement is within desired range with tolerance.
        
        Args:
            job_years: Required years of experience for the job
            desired_years: Desired years of experience to match
            tolerance_years: Number of years tolerance (above or below desired)
            
        Returns:
            Boolean indicating if experience requirement matches
        """
        if job_years is None or desired_years is None:
            # If either is not specified, consider it a match
            return True
            
        lower_bound = max(0, desired_years - tolerance_years)
        upper_bound = desired_years + tolerance_years
        
        return lower_bound <= job_years <= upper_bound
        
    def process_job_descriptions(self, category: str = None, company_name: str = None, 
                               search: str = None, location: str = None, level: str = None,
                               page: int = 1, limit: int = 20,
                               salary_min: Optional[float] = None, salary_max: Optional[float] = None,
                               experience_years: Optional[int] = None,
                               location_max_distance_km: float = 50.0,
                               salary_tolerance_percent: float = 15.0,
                               experience_tolerance_years: int = 2) -> List[Dict]:
        """
        Process job descriptions from The Muse API with fuzzy range matching.
        
        Args:
            category: Job category
            company_name: Company name filter
            search: Search term
            location: Job location filter
            level: Experience level filter
            page: Page number (1-indexed)
            limit: Maximum number of results to return
            salary_min: Minimum desired salary
            salary_max: Maximum desired salary
            experience_years: Desired years of experience
            location_max_distance_km: Maximum distance in km for location matching
            salary_tolerance_percent: Tolerance percentage for salary matching
            experience_tolerance_years: Tolerance in years for experience matching
            
        Returns:
            List of dictionaries containing processed job information
        """
        results = []
        current_page = page
        seen_jobs = set() 
    
        total_jobs_scanned = 0

        stemmer = PorterStemmer()
        search_stem = stemmer.stem(search.lower()) if search else None

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

                total_jobs_scanned += 1
                if total_jobs_scanned >= 4 * limit:
                    break
                
                if search_stem:
                    contents = job.get("contents", "")
                    title = job.get("name", "").strip()
                    cleaned_desc = self.clean_text(contents).strip()

                    stemmed_title = ' '.join([stemmer.stem(w) for w in title.lower().split()])
                    stemmed_desc = ' '.join([stemmer.stem(w) for w in cleaned_desc.lower().split()])

                    if search_stem and (search_stem not in stemmed_title and search_stem not in stemmed_desc):
                        continue
                
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
                location_str = ", ".join(location_names) if location_names else None
                category_names = [cat.get("name", "") for cat in categories]
                level_names = [level.get("name", "") for level in levels]
                
                # Apply fuzzy range filters
                # Check location range if specified
                if location and location_str:
                    location_matches = False
                    for loc_name in location_names:
                        if self.is_within_location_range(loc_name, location, location_max_distance_km):
                            location_matches = True
                            break
                    
                    # Skip if location doesn't match within range
                    if not location_matches and location_type != "Remote":
                        continue
                
                # Check salary range if specified
                if not self.is_within_salary_range(min_salary, max_salary, salary_min, salary_max, salary_tolerance_percent):
                    continue
                    
                # Check experience range if specified
                if not self.is_within_experience_range(experience, experience_years, experience_tolerance_years):
                    continue
                
                # Calculate distances for each location
                distances = {}
                if location and location_names:
                    for loc_name in location_names:
                        dist = self.calculate_location_distance(loc_name, location)
                        if dist is not None:
                            distances[loc_name] = dist
                
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
                    "locations": location_str,
                    "location_type": location_type,
                    "location_distances": distances if distances else None,
                    "levels": ", ".join(level_names) if level_names else None,
                    "publication_date": job.get("publication_date"),
                    "description": self.clean_text(contents)
                }
                results.append(job_info)
            current_page += 1
            if total_jobs_scanned >= 4 * limit:
                break
        
        print("number of job found:", len(results))
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

    def match_resume_to_jobs(self, resume_text: str, jobs: List[Dict], top_n: int = 5,
                               target_location: str = None, target_salary: float = None, 
                               target_experience: int = None, weight_content: float = 0.6,
                               weight_location: float = 0.15, weight_salary: float = 0.15,
                               weight_experience: float = 0.1) -> List[Dict]:
        """
        Match resume to jobs using text embeddings and cosine similarity with weight decay based on distance from targets.
        
        Args:
            resume_text: Text content of the resume
            jobs: List of job dictionaries
            top_n: Number of top matches to return
            target_location: Target location for distance-based weight decay
            target_salary: Target salary for salary-based weight decay (midpoint of desired range)
            target_experience: Target experience years for experience-based weight decay
            weight_content: Weight for content similarity (0.0-1.0)
            weight_location: Weight for location proximity (0.0-1.0)
            weight_salary: Weight for salary match (0.0-1.0)
            weight_experience: Weight for experience match (0.0-1.0)
            
        Returns:
            List of top matching jobs with aggregated scores
        """
        # Ensure weights sum to 1.0
        total_weight = weight_content + weight_location + weight_salary + weight_experience
        if abs(total_weight - 1.0) > 0.001:
            weight_content = weight_content / total_weight
            weight_location = weight_location / total_weight
            weight_salary = weight_salary / total_weight
            weight_experience = weight_experience / total_weight
        
        # Calculate content similarity scores
        resume_summary = self.summarize_resume(resume_text)
        corpus = [resume_summary] + [self.preprocess_job_description(job["description"]) for job in jobs]
        embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True)

        resume_embedding = embeddings[0].cpu().numpy()
        job_embeddings = embeddings[1:].cpu().numpy()

        cosine_sim = (resume_embedding @ job_embeddings.T) / (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embeddings, axis=1))

        for idx, job in enumerate(jobs):
            # Store content similarity score
            job["content_similarity"] = float(cosine_sim[idx])
            
            # Initialize component scores
            location_score = 1.0
            salary_score = 1.0
            experience_score = 1.0
            
            # Calculate location score with distance-based decay
            if target_location and job.get("locations"):
                # Start with a default low score
                location_score = 0.1
                
                # Check if remote (remote jobs get high score regardless of target location)
                if job.get("location_type") == "Remote":
                    location_score = 0.95
                # Calculate distance-based decay for each location
                elif job.get("location_distances"):
                    # Find minimum distance from any job location to target
                    min_distance = float('inf')
                    for _, distance in job["location_distances"].items():
                        min_distance = min(min_distance, distance)
                    
                    if min_distance < float('inf'):
                        # Exponential decay function: score = exp(-distance/decay_factor)
                        decay_factor = 50.0
                        location_score = max(0.1, np.exp(-min_distance / decay_factor))
            
            # Calculate salary score with salary difference decay
            if target_salary and (job.get("min_salary") is not None or job.get("max_salary") is not None):
                job_min = job.get("min_salary")
                job_max = job.get("max_salary")
                
                # Calculate job salary midpoint
                if job_min is not None and job_max is not None:
                    job_salary = (job_min + job_max) / 2
                elif job_min is not None:
                    job_salary = job_min * 1.1 
                elif job_max is not None:
                    job_salary = job_max * 0.9 
                else:
                    job_salary = None
                
                if job_salary is not None:
                    # Calculate percentage difference
                    percent_diff = abs(job_salary - target_salary) / target_salary
                    decay_factor = 0.15 
                    salary_score = max(0.1, np.exp(-percent_diff / decay_factor))
            
            # Calculate experience score with years difference decay
            if target_experience is not None and job.get("years_experience") is not None:
                years_diff = abs(job.get("years_experience") - target_experience)
                decay_factor = 2.0 
                experience_score = max(0.1, np.exp(-years_diff / decay_factor))
            
            # Calculate weighted aggregate score
            aggregate_score = (
                weight_content * job["content_similarity"] +
                weight_location * location_score +
                weight_salary * salary_score +
                weight_experience * experience_score
            )
            
            job["location_score"] = location_score
            job["salary_score"] = salary_score
            job["experience_score"] = experience_score
            job["similarity_score"] = aggregate_score

        sorted_jobs = sorted(jobs, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_jobs[:top_n]

    def match_resume_with_fuzzy_criteria(self, resume_text: str, 
                                       location: str = None, 
                                       salary_min: Optional[float] = None, 
                                       salary_max: Optional[float] = None,
                                       experience_years: Optional[int] = None,
                                       category: str = None, 
                                       company_name: str = None,
                                       search: str = None, 
                                       level: str = None,
                                       location_max_distance_km: float = 50.0,
                                       salary_tolerance_percent: float = 15.0,
                                       experience_tolerance_years: int = 2,
                                       limit: int = 20,
                                       top_n: int = 5,
                                       weight_content: float = 0.6,
                                       weight_location: float = 0.15,
                                       weight_salary: float = 0.15,
                                       weight_experience: float = 0.1) -> List[Dict]:
        """
        Comprehensive method to match a resume to jobs with fuzzy criteria and weighted scoring.
        
        Args:
            resume_text: Text content of the resume
            location: Desired job location
            salary_min: Minimum desired salary
            salary_max: Maximum desired salary
            experience_years: Years of experience the candidate has
            category: Job category filter
            company_name: Company name filter
            search: Search term
            level: Experience level filter
            location_max_distance_km: Maximum distance for location matching
            salary_tolerance_percent: Percentage tolerance for salary matching
            experience_tolerance_years: Years tolerance for experience matching
            limit: Maximum number of results to fetch
            top_n: Number of top matches to return
            weight_content: Weight for content similarity (0.0-1.0)
            weight_location: Weight for location proximity (0.0-1.0)
            weight_salary: Weight for salary match (0.0-1.0)
            weight_experience: Weight for experience match (0.0-1.0)
            
        Returns:
            List of top matching jobs with weighted similarity scores
        """
        # Get jobs with fuzzy criteria matching
        jobs = self.process_job_descriptions(
            category=category,
            company_name=company_name,
            search=search,
            location=location,
            level=level,
            limit=limit,
            salary_min=salary_min,
            salary_max=salary_max,
            experience_years=experience_years,
            location_max_distance_km=location_max_distance_km,
            salary_tolerance_percent=salary_tolerance_percent,
            experience_tolerance_years=experience_tolerance_years
        )
        
        # Calculate target salary
        target_salary = None
        if salary_min is not None or salary_max is not None:
            if salary_min is not None and salary_max is not None:
                target_salary = (salary_min + salary_max) / 2
            elif salary_min is not None:
                target_salary = salary_min * 1.15  
            elif salary_max is not None:
                target_salary = salary_max * 0.85
        
        # Match the resume to these pre-filtered jobs with weighted scoring
        return self.match_resume_to_jobs(
            resume_text=resume_text, 
            jobs=jobs, 
            top_n=top_n,
            target_location=location,
            target_salary=target_salary,
            target_experience=experience_years,
            weight_content=weight_content,
            weight_location=weight_location,
            weight_salary=weight_salary,
            weight_experience=weight_experience
        )

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

