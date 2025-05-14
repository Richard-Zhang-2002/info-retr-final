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

#extract jobs, general class
class MuseJobExtractor:
    def __init__(self, api_key=None):
        
        self.api_url = "https://www.themuse.com/api/public/jobs"
        self.api_key = api_key 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        try:
            with open("openai_key.txt", "r") as f:
                key = f.read().strip()
                self.client = Groq(api_key=key)
        except FileNotFoundError:
            print("Warning: openai_key.txt not found. Resume summarization will fail.")


    #basically convert the input location into a fixed format, for comparison later
    def normalize_location_with_ai(self, location_query: str) -> Optional[Dict[str, str]]:
        prompt = (
            f"Given the location '{location_query}', return a JSON object with available structured fields: "
            f"'city', 'state' (use 2-letter U.S. state abbreviations if applicable), and 'country'. "
            f"Only include fields that are confidently applicable. "
            f"For example, 'Tokyo' → {{'city': 'Tokyo', 'country': 'Japan'}}; 'Baltimore' → {{'city': 'Baltimore', 'state': 'MD', 'country': 'USA'}}. "
            f"Use lowercase keys. Format the output as valid JSON."
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```") and content.endswith("```"):
                content = "\n".join(content.splitlines()[1:-1]).strip()
            print(f"[AI raw response] {content}")
            parsed = json.loads(content)
            return {k: v.lower() for k, v in parsed.items() if v is not None}

        except Exception as e:
            print(f"AI normalization failed: {e}")
            return None

            
    #different weighting for different level matches, same city would be ideal, same state may be fine, same country is too far...remote is always mediocure though
    def compute_location_score(self, user_loc: Dict[str, str], job_locs: List[Dict[str, str]]) -> float:
        max_score = 0.1
        for job in job_locs:
            if job.get("remote"):
                max_score = max(max_score, 0.7)
            elif user_loc.get("city") and job.get("city") and user_loc["city"] == job["city"]:
                max_score = max(max_score, 1.0)
            elif user_loc.get("state") and job.get("state") and user_loc["state"] == job["state"]:
                max_score = max(max_score, 0.85)
            elif user_loc.get("country") and job.get("country") and user_loc["country"] == job["country"]:
                max_score = max(max_score, 0.85)
        return max_score


    def fetch_jobs(self, category: str = None, company_name: str = None, 
                  search: str = None, location: str = None, level: str = None,
                  page: int = 1, descending: bool = False) -> Dict[str, Any]:
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
            
            #limit = response.headers.get('X-RateLimit-Limit')
            #remaining = response.headers.get('X-RateLimit-Remaining')
            #reset = response.headers.get('X-RateLimit-Reset')
            #print(f"Rate limits - Total: {limit}, Remaining: {remaining}, Reset in: {reset}s")
            
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
        if not contents:
            return None
            
        clean_contents = re.sub(r'<[^>]+>', ' ', contents)
        
        #common salary patterns
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
        if not contents:
            return None
            
        clean_contents = re.sub(r'<[^>]+>', ' ', contents).lower()
        
        location_names = [loc.get('name', '').lower() for loc in locations if loc.get('name')]
        
        for loc in location_names:
            if 'remote' in loc:
                return "Remote"
            if 'hybrid' in loc:
                return "Hybrid"
        
        work_arrangements = [
            (r'\bremote\b', "Remote"),
            (r'\bhybrid\b', "Hybrid"),
            (r'\bon[-\s]site\b', "On-site"),
            (r'\bin[-\s]office\b', "In-office")
        ]
        
        for pattern, arrangement in work_arrangements:
            if re.search(pattern, clean_contents, re.IGNORECASE):
                return arrangement
                
        if locations and len(locations) > 0:
            return locations[0].get('name')
            
        return None
    
    def is_within_salary_range(self, job_min: Optional[float], job_max: Optional[float], 
                              desired_min: Optional[float], desired_max: Optional[float], 
                              tolerance_percent: float = 15.0) -> bool:
        if job_min is None and job_max is None:
            #no salary info, assume it matches
            return True
            
        if desired_min is None and desired_max is None:
            #didnt ask for a salary, so any job is within range
            return True
            
        if desired_min is not None:
            min_with_tolerance = desired_min * (1 - tolerance_percent / 100)
        else:
            min_with_tolerance = float('-inf')
        
        if desired_max is not None:
            max_with_tolerance = desired_max * (1 + tolerance_percent / 100)
        else:
            max_with_tolerance = float('inf')
            
        if job_min is not None and job_max is not None:
            return not (job_max < min_with_tolerance or job_min > max_with_tolerance)
        elif job_min is not None:
            return job_min <= max_with_tolerance
        elif job_max is not None:
            return job_max >= min_with_tolerance
            
        return False
        
    def is_within_experience_range(self, job_years: Optional[int], desired_years: Optional[int], 
                                 tolerance_years: int = 2) -> bool:
        
        if job_years is None or desired_years is None:
            return True
            
        lower_bound = max(0, desired_years - tolerance_years)
        upper_bound = desired_years + tolerance_years
        
        return lower_bound <= job_years <= upper_bound
        
    def process_job_descriptions(self, category: str = None, company_name: str = None, 
                               search: str = None, location: str = None, level: str = None,
                               page: int = 1, limit: int = 20,
                               salary_min: Optional[float] = None, salary_max: Optional[float] = None,
                               experience_years: Optional[int] = None,
                               location_max_distance: float = 2000,
                               salary_tolerance_percent: float = 100,
                               experience_tolerance_years: int = 20,
                               allow_remote_jobs: bool = True) -> List[Dict]:
        results = []
        current_page = page
        seen_jobs = set() 
    
        total_jobs_scanned = 0

        stemmer = PorterStemmer()
        search_stem = stemmer.stem(search.lower()) if search else None

        user_loc_dict = None
        if location:
            user_loc_dict = self.normalize_location_with_ai(location)

        while len(results) < limit:
            response = self.fetch_jobs(category, company_name, search, location, level, page=current_page)
            jobs = response.get("results", [])
            if not jobs:
                break
                
            for job in jobs:
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
                
                locations = job.get("locations", [])
                categories = job.get("categories", [])
                levels = job.get("levels", [])
                
                salary_range = self.extract_salary_range(contents)
                experience = self.extract_experience(contents)
                location_type = self.extract_location_type(locations, contents)
                
                min_salary, max_salary = (None, None)
                if salary_range:
                    min_salary, max_salary = salary_range
                
                location_names = [loc.get("name", "") for loc in locations]

                #if we dont allow remote jobs, dont add them
                if not allow_remote_jobs:
                    is_remote_only = all("remote" in loc.lower() for loc in location_names)
                    if is_remote_only:
                        continue

                location_str = ", ".join(location_names) if location_names else None
                #print("location:",location_str)
                #for name in location_names:
                #    print(name)
                category_names = [cat.get("name", "") for cat in categories]
                level_names = [level.get("name", "") for level in levels]
                
                normalized_locations = []

                for loc in location_names:
                    parts = [p.strip() for p in loc.split(",")]
                    loc_obj = {}

                    if "remote" in loc.lower():
                        loc_obj["remote"] = True

                    else:
                        #should always be the case
                        if len(parts) == 2:
                            city_part, second_part = parts
                            if re.fullmatch(r"[A-Z]{2}", second_part):
                                loc_obj = {
                                    "city": city_part.lower(),
                                    "state": second_part.lower(),
                                }
                            else:
                                loc_obj = {
                                    "city": city_part.lower(),
                                    "country": second_part.lower()
                                }
                        
                        elif len(parts) == 1:
                            #handle cases like singapore, both a city and a country
                            loc_obj = {
                                "city": parts[0].lower(),
                                "country": parts[0].lower()
                            }

                        else:
                            print("WHAT? We have something other than two??")
                            print(f"Unexpected location format: '{loc}' with parts = {parts}")



                    normalized_locations.append(loc_obj)


                if user_loc_dict is not None:
                    matches = False
                    for job_loc in normalized_locations:
                        if (
                            ("city" in user_loc_dict and "city" in job_loc and user_loc_dict["city"] == job_loc["city"]) or
                            ("state" in user_loc_dict and "state" in job_loc and user_loc_dict["state"] == job_loc["state"]) or
                            ("country" in user_loc_dict and "country" in job_loc and user_loc_dict["country"] == job_loc["country"])
                        ):
                            matches = True
                            break
                    if not matches:
                        if allow_remote_jobs and any(loc.get("remote") for loc in normalized_locations):
                            pass#remote can be any location
                        else:
                            continue

                
                if not self.is_within_salary_range(min_salary, max_salary, salary_min, salary_max, salary_tolerance_percent):
                    continue
                    
                if not self.is_within_experience_range(experience, experience_years, experience_tolerance_years):
                    continue
                
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
                    "location_normalized":normalized_locations,
                    "levels": ", ".join(level_names) if level_names else None,
                    "publication_date": job.get("publication_date"),
                    "description": self.clean_text(contents)
                }
                results.append(job_info)
            current_page += 1
            if total_jobs_scanned >= 4 * limit:
                break
        
        print("Number of job found:", len(results))
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

    #job experiences and salary are not absolute clear cut: I am looking a job that gives me 10k, 9k isnt ideal, but I am willing to take a look at it
    def match_resume_to_jobs(self, resume_text: str, jobs: List[Dict], top_n: int = 5,
                               target_location: str = None, target_salary: float = None, 
                               target_experience: int = None, weight_content: float = 0.6,
                               weight_location: float = 0.15, weight_salary: float = 0.15,
                               weight_experience: float = 0.1) -> List[Dict]:
        #normalize weightsum
        total_weight = weight_content + weight_location + weight_salary + weight_experience
        if abs(total_weight - 1.0) > 0.001:
            weight_content = weight_content / total_weight
            weight_location = weight_location / total_weight
            weight_salary = weight_salary / total_weight
            weight_experience = weight_experience / total_weight
        
        resume_summary = self.summarize_resume(resume_text)
        corpus = [resume_summary] + [self.preprocess_job_description(job["description"]) for job in jobs]
        embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True)

        resume_embedding = embeddings[0].cpu().numpy()
        job_embeddings = embeddings[1:].cpu().numpy()

        cosine_sim = (resume_embedding @ job_embeddings.T) / (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embeddings, axis=1))

        user_loc_dict = self.normalize_location_with_ai(target_location) if target_location else {}

        for idx, job in enumerate(jobs):
            job["content_similarity"] = float(cosine_sim[idx])
            
            location_score = 1.0
            salary_score = 1.0
            experience_score = 1.0
            
            if user_loc_dict and job.get("location_normalized"):
                location_score = self.compute_location_score(user_loc_dict, job["location_normalized"])
            
            if target_salary and (job.get("min_salary") is not None or job.get("max_salary") is not None):
                job_min = job.get("min_salary")
                job_max = job.get("max_salary")
                
                if job_min is not None and job_max is not None:
                    job_salary = (job_min + job_max) / 2
                elif job_min is not None:
                    job_salary = job_min * 1.1 
                elif job_max is not None:
                    job_salary = job_max * 0.9 
                else:
                    job_salary = None
                
                if job_salary is not None:
                    percent_diff = abs(job_salary - target_salary) / target_salary
                    decay_factor = 0.15 
                    salary_score = max(0.1, np.exp(-percent_diff / decay_factor))
            
            if target_experience is not None and job.get("years_experience") is not None:
                years_diff = abs(job.get("years_experience") - target_experience)
                decay_factor = 2.0 
                experience_score = max(0.1, np.exp(-years_diff / decay_factor))
            
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
                                       location_max_distance: float = 2000,
                                       salary_tolerance_percent: float = 100,
                                       experience_tolerance_years: int = 20,
                                       limit: int = 500,
                                       top_n: int = 100,
                                       weight_content: float = 0.6,
                                       weight_location: float = 0.15,
                                       weight_salary: float = 0.15,
                                       weight_experience: float = 0.1,
                                       allow_remote_jobs: bool = True) -> List[Dict]:
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
            location_max_distance=location_max_distance,
            salary_tolerance_percent=salary_tolerance_percent,
            experience_tolerance_years=experience_tolerance_years,
            allow_remote_jobs=allow_remote_jobs
        )
        
        target_salary = None
        if salary_min is not None or salary_max is not None:
            if salary_min is not None and salary_max is not None:
                target_salary = (salary_min + salary_max) / 2
            elif salary_min is not None:
                target_salary = salary_min * 1.15  
            elif salary_max is not None:
                target_salary = salary_max * 0.85
        
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
