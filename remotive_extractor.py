import requests
import re
import json
import csv
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

class RemotiveJobExtractor:
    """Class to fetch remote job descriptions from Remotive API and extract key information."""
    
    def __init__(self):
        """Initialize with Remotive API URL."""
        self.api_url = "https://remotive.com/api/remote-jobs"
    
    def fetch_jobs(self, 
                  category: str = None, 
                  company_name: str = None, 
                  search: str = None, 
                  limit: int = None) -> Dict[str, Any]:
        """
        Fetch job listings from the Remotive API.
        
        Args:
            category: Filter by job category
            company_name: Filter by company name
            search: Search term for job title and description
            limit: Maximum number of jobs to return
            
        Returns:
            Dictionary containing job count and list of job listings
        """
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
            return {"job-count": X0, "jobs": []}
    
    def extract_salary_range(self, salary_text: str, description: str) -> Optional[Tuple[float, float]]:
        """
        Extract salary range from both salary field and job description text.
        
        Args:
            salary_text: Dedicated salary field from API
            description: Full job description text
            
        Returns:
            Tuple of (min_salary, max_salary) or None if not found
        """
        # First try the dedicated salary field
        if salary_text and isinstance(salary_text, str):
            # Pattern matches common salary formats in the salary field
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
                        
                        # Convert if salary is in 'k' format
                        if 'k' in salary_text.lower():
                            min_salary *= 1000
                            max_salary *= 1000
                            
                        return (min_salary, max_salary)
                    except (ValueError, IndexError):
                        continue
        
        # If salary field didn't work, try extracting from description
        if description:
            # Strip HTML tags for better text processing
            clean_description = re.sub(r'<[^>]+>', ' ', description)
            
            # Pattern matches common salary formats in descriptions
            patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]\s*[-–to]+\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[k]',
                r'salary\s*(?:range)?:\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                r'annual\s*salary\s*(?:range)?:\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–to]+\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, clean_description, re.IGNORECASE)
                if matches:
                    try:
                        min_salary = float(matches.group(1).replace(',', ''))
                        max_salary = float(matches.group(2).replace(',', ''))
                        
                        # Convert if salary is in 'k' format
                        if 'k' in clean_description[matches.start():matches.end()].lower():
                            min_salary *= 1000
                            max_salary *= 1000
                            
                        return (min_salary, max_salary)
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def extract_experience(self, description: str) -> Optional[int]:
        """
        Extract years of experience required from job description.
        
        Args:
            description: Job description text
            
        Returns:
            Integer representing years of experience or None if not found
        """
        if not description:
            return None
            
        # Strip HTML tags for better text processing
        clean_description = re.sub(r'<[^>]+>', ' ', description)
        
        patterns = [
            r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience|exp)',
            r'(\d+)\+?\s*(?:year|yr)s?\s*experience',
            r'experience:\s*(\d+)\+?\s*(?:year|yr)s?',
            r'minimum\s*(?:of)?\s*(\d+)\s*(?:year|yr)s?\s*(?:of)?\s*experience',
            r'at\s*least\s*(\d+)\s*(?:year|yr)s?\s*(?:of)?\s*experience'
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
        """
        Extract work location from both required location field and job description.
        
        Args:
            required_location: Candidate required location field from API
            description: Full job description text
            
        Returns:
            String representing location or None if not found
        """
        # First check the dedicated location field
        if required_location and isinstance(required_location, str) and required_location.strip() != "":
            return required_location.strip()
        
        if not description:
            return None
            
        # Strip HTML tags for better text processing
        clean_description = re.sub(r'<[^>]+>', ' ', description)
        
        # Try to match location patterns in description
        patterns = [
            r'location:\s*(.*?)(?:\.|,|\n)',
            r'based\s*in\s*((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))',
            r'position\s*(?:is\s+)?located\s*in\s*((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))',
            r'(?:remote|hybrid|on-site|in-office)\s*(?:position|job|role|work)?\s*(?:in|at)?\s*((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, clean_description, re.IGNORECASE)
            if matches:
                try:
                    return matches.group(1).strip()
                except (IndexError):
                    continue
        
        # Check for specific work arrangements
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
    
    def process_job_descriptions(self, 
                                category: str = None, 
                                company_name: str = None, 
                                search: str = None, 
                                limit: int = None) -> List[Dict]:
        """
        Process job descriptions to extract key information.
        
        Args:
            category: Filter by job category
            company_name: Filter by company name
            search: Search term for job title and description
            limit: Maximum number of jobs to return
            
        Returns:
            List of dictionaries with extracted job information
        """
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
            
            # Prepare salary range values for output
            min_salary = None
            max_salary = None
            if salary_range:
                min_salary = salary_range[0]
                max_salary = salary_range[1]
            
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
                "description": description
            }
            results.append(job_info)
        
        return results
    
    def save_json(self, results: List[Dict], filename: str = None) -> str:
        """
        Save processed job information to a JSON file.
        
        Args:
            results: List of job information dictionaries
            filename: Optional filename to save results
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"remotive_job_data_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        return os.path.abspath(filename)
    
    def save_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        Save processed job information to a CSV file.
        
        Args:
            results: List of job information dictionaries
            filename: Optional filename to save results
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"remotive_job_data_{timestamp}.csv"
        
        if not results:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                f.write("No job listings found")
            return os.path.abspath(filename)
        
        # Get all unique keys from all dictionaries
        fieldnames = set()
        for job in results:
            fieldnames.update(job.keys())
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(results)
        
        return os.path.abspath(filename)
    
    def save_excel(self, results: List[Dict], filename: str = None) -> str:
        """
        Save processed job information to an Excel file.
        
        Args:
            results: List of job information dictionaries
            filename: Optional filename to save results
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"remotive_job_data_{timestamp}.xlsx"
        
        # Convert to DataFrame and save to Excel
        df = pd.DataFrame(results)
        df.to_excel(filename, index=False)
        
        return os.path.abspath(filename)
    
    def save_results(self, results: List[Dict], format: str = "json", filename: str = None) -> str:
        """
        Save processed job information to a file in the specified format.
        
        Args:
            results: List of job information dictionaries
            format: File format to save as ("json", "csv", or "excel")
            filename: Optional filename to save results
            
        Returns:
            Path to the saved file
        """
        format = format.lower()
        
        if format == "json":
            return self.save_json(results, filename)
        elif format == "csv":
            return self.save_csv(results, filename)
        elif format == "excel":
            return self.save_excel(results, filename)
        else:
            raise ValueError(f"Unsupported file format: {format}. Use 'json', 'csv', or 'excel'.")

def main():
    """Main function to run the job extractor."""
    extractor = RemotiveJobExtractor()
    
    # Get user input for search parameters
    print("\n=== Remotive Job Extractor ===")
    print("Leave fields blank and press Enter to skip them.")
    
    category = input("Enter job category (e.g., 'software-dev'): ").strip() or None
    company = input("Enter company name (optional): ").strip() or None
    search_term = input("Enter search term (optional): ").strip() or None
    
    try:
        limit_input = input("Enter maximum number of jobs (optional): ").strip()
        limit = int(limit_input) if limit_input else None
    except ValueError:
        print("Invalid limit, using no limit.")
        limit = None
    
    print("\nFetching and processing jobs...")
    results = extractor.process_job_descriptions(
        category=category,
        company_name=company,
        search=search_term,
        limit=limit
    )
    
    if not results:
        print("No jobs found matching your criteria.")
        return
    
    print(f"\nFound {len(results)} jobs matching your criteria.")
    
    # Choose file format
    print("\nAvailable file formats:")
    print("1. JSON (default)")
    print("2. CSV")
    print("3. Excel (.xlsx)")
    
    format_choice = input("Enter format number (1-3): ").strip() or "1"
    
    format_map = {"1": "json", "2": "csv", "3": "excel"}
    file_format = format_map.get(format_choice, "json")
    
    # Allow custom filename
    custom_filename = input(f"Enter filename (or press Enter for default): ").strip() or None
    
    # Save results
    output_file = extractor.save_results(results, format=file_format, filename=custom_filename)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print sample of results
    if results:
        print("\nSample of extracted information:")
        for job in results[:2]:  # Show first 2 jobs as sample
            print(f"\nJob: {job['title']} at {job['company']}")
            print(f"URL: {job['url']}")
            
            if job.get('min_salary') and job.get('max_salary'):
                print(f"Salary Range: ${job['min_salary']} - ${job['max_salary']}")
            else:
                print(f"Salary Range: Not found")
                
            if job.get('years_experience'):
                print(f"Experience Required: {job['years_experience']} years")
            else:
                print(f"Experience Required: Not specified")
                
            print(f"Location: {job.get('location', 'Not specified')}")

# Run the script if executed directly
if __name__ == "__main__":
    main()