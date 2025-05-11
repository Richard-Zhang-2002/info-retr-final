import customtkinter as ctk
from tkinter import filedialog
import fitz
import webbrowser
from muse_extractor import MuseJobExtractor
import re


extractor = MuseJobExtractor()  # Initialize without API key - consider adding API key for production use

# Setting UI related stuff
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("Resume + Job Matcher")
app.geometry("900x600")

# Save resume path (uploaded resume)
resume_path = ctk.StringVar()
# The job the user clicked on (to view detail)
current_selected_title = None
# Access the job by its title (here we assume no duplicate job titles)
job_card_refs = {}

# Simple summarization function - not really AI but helps with large text blocks
def summarize_text(text):
    if not text:
        return "No description available."
    
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Replace multiple newlines or spaces with single
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim leading/trailing spaces
    text = text.strip()

    # If still too short after cleaning
    if len(text) < 30:
        return "No substantial description available."

    return text

# Check if the input includes only digits
def only_digits(input):
    return input.isdigit() or input == ""

# Check if the input is a valid float number
def validate_float(input):
    # Allow empty string for deletion
    if input == "":
        return True
    
    # Allow single decimal point
    if input == ".":
        return True
    
    # Allow digits and one decimal point
    try:
        float(input)
        return True
    except ValueError:
        return False

# Happens when user clicks on "upload resume(pdf)"
def upload_resume():
    path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if path:
        resume_path.set(path)

# Use fitz to read all text in pdf (no pictures)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# Navigate to job URL
def open_job_url(url):
    webbrowser.open_new(url)

# Happens when the user clicks on one of the searched jobs
def toggle_description(title, description, salary, location, years, category, job_type, levels, card_frame):
    global current_selected_title

    # If we already selected a title, unselect it (do not show its description)
    if current_selected_title == title:
        current_selected_title = None
        description_frame.grid_remove()
        card_frame.configure(fg_color="#f1f1f1")
        return

    # Otherwise we show the selected job info
    current_selected_title = title
    for card in job_card_refs.values():
        card.configure(fg_color="#f1f1f1")  # white for unselected
    card_frame.configure(fg_color="#dbeafe")  # light blue for selected

    # Populate the description based on input (aka based on job info)
    job_title_label.configure(text=title)
    salary_label.configure(text=f"💵 Salary: {salary or 'N/A'}")
    location_label.configure(text=f"📍 Location: {location or 'N/A'}")
    experience_label.configure(text=f"📆 Required Experience: {years or 'N/A'} years")
    category_label.configure(text=f"🧩 Category: {category or 'N/A'}")
    type_label.configure(text=f"🌎 Job Type: {job_type or 'N/A'}")
    level_label.configure(text=f"🔝 Level: {levels or 'N/A'}")
    summary_label.configure(text=description or "No description available.")

    # Make a grid to place these things
    description_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))

# Create job in UI (happens after I click submit and get all sort of job information)
def add_job(title, url, description, salary, location, years, category, job_type, levels):
    # Create the box for this job and tie it with the specific title
    job_card = ctk.CTkFrame(job_list_frame, fg_color="#f1f1f1", cursor="hand2")
    job_card.pack(fill="x", padx=5, pady=5)
    job_card_refs[title] = job_card

    # Second layer, for the padding effect
    card_body = ctk.CTkFrame(job_card, fg_color="transparent")
    card_body.pack(fill="x", padx=5, pady=5)

    # Title and button of this job
    title_label = ctk.CTkLabel(card_body, text=title, font=("Segoe UI", 12, "bold"), anchor="w", width=280)
    title_label.pack(side="left")

    # If I click on the button, I should be navigated to the correlated url
    ctk.CTkButton(card_body, text="Navigate →", width=80, command=lambda: open_job_url(url)).pack(side="right")

    # Basically if I click anywhere besides the button, I should activate the description of this job
    for i in [job_card, card_body, title_label]:
        i.bind("<Button-1>", lambda e: toggle_description(title, description, salary, location, years, category, job_type, levels, job_card))

# Happens when the Submit button is clicked
def submit():
    # Clear the jobs in the list (as we are loading new ones)
    global job_card_refs
    job_card_refs = {}

    # Destroy the correlated job UI items too
    for i in job_list_frame.winfo_children():
        i.destroy()
    
    # Nothing is selected, obviously
    global current_selected_title
    current_selected_title = None

    # There shouldn't be any description too, as the jobs are brand new (and are unselected)
    description_frame.grid_remove()

    # Get filter values
    min_salary = salary_min.get().strip()
    max_salary = salary_max.get().strip() 
    years_exp = experience.get().strip()
    desired_loc = location.get().strip()
    search_keyword = keyword.get().strip()
    
    # Convert inputs to appropriate types for the extractor
    min_salary_val = float(min_salary) if min_salary else None
    max_salary_val = float(max_salary) if max_salary else None
    years_exp_val = int(years_exp) if years_exp else None
    
    # Get resume
    resume = resume_path.get()
    if not resume:
        print("Please upload a resume!")
        status_label.configure(text="Please upload a resume!")
        return

    status_label.configure(text="Loading jobs... Please wait.")
    app.update_idletasks()  # Update UI to show status message

    # Get resume text
    try:
        resume_text = extractor.load_resume(resume)
    except Exception as e:
        status_label.configure(text=f"Error loading resume: {str(e)}")
        return

    # Get the weights from the UI
    try:
        content_weight = float(content_weight_entry.get().strip()) if content_weight_entry.get().strip() else 0.6
        location_weight = float(location_weight_entry.get().strip()) if location_weight_entry.get().strip() else 0.15
        salary_weight = float(salary_weight_entry.get().strip()) if salary_weight_entry.get().strip() else 0.15
        experience_weight = float(experience_weight_entry.get().strip()) if experience_weight_entry.get().strip() else 0.1
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = content_weight + location_weight + salary_weight + experience_weight
        if total_weight <= 0:
            raise ValueError("Total weight must be greater than 0")
            
        content_weight /= total_weight
        location_weight /= total_weight
        salary_weight /= total_weight
        experience_weight /= total_weight
        
    except ValueError:
        status_label.configure(text="Invalid weight values. Using defaults.")
        content_weight = 0.6
        location_weight = 0.15
        salary_weight = 0.15
        experience_weight = 0.1
    
    # Get level based on years of experience
    level = None
    if years_exp_val is not None:
        if years_exp_val <= 1:
            level = "entry"
        elif years_exp_val <= 5:
            level = "mid"
        else:
            level = "senior"
    
    try:
        # Use the more comprehensive method from the extractor
        matched_jobs = extractor.match_resume_with_fuzzy_criteria(
            resume_text=resume_text,
            location=desired_loc if desired_loc else None,
            salary_min=min_salary_val,
            salary_max=max_salary_val,
            experience_years=years_exp_val,
            search=search_keyword if search_keyword else None,
            level=level,
            location_max_distance=2000, 
            salary_tolerance_percent=100, 
            experience_tolerance_years=20,  
            limit=100,  # Fetch up to 100 jobs before filtering
            top_n=20,   # Return top 20 matches
            weight_content=content_weight,
            weight_location=location_weight,
            weight_salary=salary_weight,
            weight_experience=experience_weight
        )
    except Exception as e:
        status_label.configure(text=f"Error matching jobs: {str(e)}")
        print(f"Error details: {e}")
        return
    
    if not matched_jobs:
        status_label.configure(text="No matching jobs found. Try broadening your search criteria.")
        return
    
    # Add jobs to UI
    for job in matched_jobs:
        # Format salary if available
        salary_display = None
        if job.get('min_salary') and job.get('max_salary'):
            salary_display = f"${job['min_salary']:.0f}–${job['max_salary']:.0f}"
        elif job.get('min_salary'):
            salary_display = f"${job['min_salary']:.0f}+"
        elif job.get('max_salary'):
            salary_display = f"Up to ${job['max_salary']:.0f}"
            
        add_job(
            title=job['title'],
            url=job['url'],
            description=summarize_text(job.get('description', "No description available.")),
            salary=salary_display,
            location=job['locations'],
            years=job['years_experience'],
            category=job['category'],
            job_type=job.get('job_type', 'N/A'),
            levels=job.get('levels', 'N/A')
        )
    
    status_label.configure(text=f"Found {len(matched_jobs)} matching jobs!")


# UI layout
# Left frame for user input and controls
left_frame = ctk.CTkFrame(app, width=250)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

validate_num = app.register(only_digits)
validate_float_cmd = app.register(validate_float)

# Create a frame to show more advanced options
advanced_frame = ctk.CTkFrame(left_frame)

# Create a toggle for advanced options
show_advanced = ctk.BooleanVar(value=False)

def toggle_advanced():
    if show_advanced.get():
        advanced_frame.pack(anchor="w", fill="x", pady=10)
    else:
        advanced_frame.pack_forget()

# Basic search fields
ctk.CTkLabel(left_frame, text="Keyword Search").pack(anchor="w")
keyword = ctk.CTkEntry(left_frame)
keyword.pack(anchor="w", fill="x", pady=(0, 10))

ctk.CTkLabel(left_frame, text="Salary Range($)").pack(anchor="w", pady=(5, 0))
salary_min = ctk.CTkEntry(left_frame, placeholder_text="Min", validate="key", validatecommand=(validate_num, "%S"))
salary_min.pack(anchor="w", fill="x")
salary_max = ctk.CTkEntry(left_frame, placeholder_text="Max", validate="key", validatecommand=(validate_num, "%S"))
salary_max.pack(anchor="w", fill="x", pady=(0, 10))

ctk.CTkLabel(left_frame, text="Years of Experience").pack(anchor="w")
experience = ctk.CTkEntry(left_frame, validate="key", validatecommand=(validate_num, "%S"))
experience.pack(anchor="w", fill="x", pady=(0, 10))

ctk.CTkLabel(left_frame, text="Desired Work Location").pack(anchor="w")
location = ctk.CTkEntry(left_frame)
location.pack(anchor="w", fill="x", pady=(0, 10))

# Advanced options toggle
ctk.CTkCheckBox(left_frame, text="Show Weight Controls", variable=show_advanced, command=toggle_advanced).pack(anchor="w", pady=(10, 0))

# Weight adjustment controls
ctk.CTkLabel(advanced_frame, text="Matching Weights", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(5, 5))

ctk.CTkLabel(advanced_frame, text="Content Relevance").pack(anchor="w")
content_weight_entry = ctk.CTkEntry(advanced_frame, validate="key", validatecommand=(validate_float_cmd, "%P"))
content_weight_entry.pack(anchor="w", fill="x", pady=(0, 5))
content_weight_entry.insert(0, "0.6")

ctk.CTkLabel(advanced_frame, text="Location Proximity").pack(anchor="w")
location_weight_entry = ctk.CTkEntry(advanced_frame, validate="key", validatecommand=(validate_float_cmd, "%P"))
location_weight_entry.pack(anchor="w", fill="x", pady=(0, 5))
location_weight_entry.insert(0, "0.15")

ctk.CTkLabel(advanced_frame, text="Salary Match").pack(anchor="w")
salary_weight_entry = ctk.CTkEntry(advanced_frame, validate="key", validatecommand=(validate_float_cmd, "%P"))
salary_weight_entry.pack(anchor="w", fill="x", pady=(0, 5))
salary_weight_entry.insert(0, "0.15")

ctk.CTkLabel(advanced_frame, text="Experience Match").pack(anchor="w")
experience_weight_entry = ctk.CTkEntry(advanced_frame, validate="key", validatecommand=(validate_float_cmd, "%P"))
experience_weight_entry.pack(anchor="w", fill="x", pady=(0, 5))
experience_weight_entry.insert(0, "0.1")

ctk.CTkLabel(advanced_frame, text="Note: Weights will be normalized", font=("Segoe UI", 10, "italic")).pack(anchor="w", pady=(0, 5))

# Resume upload and submit buttons
ctk.CTkButton(left_frame, text="Upload Resume (PDF)", command=upload_resume).pack(pady=10)
ctk.CTkLabel(left_frame, textvariable=resume_path, wraplength=250).pack()
ctk.CTkButton(left_frame, text="Submit", command=submit).pack(pady=15)

# Status label for feedback
status_label = ctk.CTkLabel(left_frame, text="", wraplength=230)
status_label.pack(pady=10)

# Right frame for job listing and job details
main_right_frame = ctk.CTkFrame(app)
main_right_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
main_right_frame.grid_rowconfigure(0, weight=3)  # job listing will take 60% of screen if description exists, else 100% (height)
main_right_frame.grid_rowconfigure(1, weight=2)  # weight 3:2
main_right_frame.grid_columnconfigure(0, weight=1)

# Job list on the top
job_list_frame_container = ctk.CTkFrame(main_right_frame)
job_list_frame_container.grid(row=0, column=0, sticky="nsew")

job_list_canvas = ctk.CTkScrollableFrame(job_list_frame_container, width=550)
job_list_canvas.pack(fill="both", expand=True, padx=10)
job_list_frame = job_list_canvas

# Description on the bottom (remove as it is not selected in the initial state)
description_frame = ctk.CTkScrollableFrame(main_right_frame, fg_color="#ffffff", corner_radius=8)
description_frame.grid_remove()

job_title_label = ctk.CTkLabel(description_frame, text="", font=("Arial", 16, "bold"), anchor="w")
job_title_label.pack(anchor="w", padx=10, pady=(5, 2))

salary_label = ctk.CTkLabel(description_frame, text="", anchor="w")
salary_label.pack(anchor="w", padx=10)

location_label = ctk.CTkLabel(description_frame, text="", anchor="w")
location_label.pack(anchor="w", padx=10)

experience_label = ctk.CTkLabel(description_frame, text="", anchor="w")
experience_label.pack(anchor="w", padx=10)

category_label = ctk.CTkLabel(description_frame, text="", anchor="w")
category_label.pack(anchor="w", padx=10)

type_label = ctk.CTkLabel(description_frame, text="", anchor="w")
type_label.pack(anchor="w", padx=10)

level_label = ctk.CTkLabel(description_frame, text="", anchor="w")
level_label.pack(anchor="w", padx=10)

summary_label = ctk.CTkLabel(description_frame, text="", anchor="w", wraplength=700, justify="left")
summary_label.pack(anchor="w", padx=10, pady=(10, 5))

# Update textbox after screen resizing
def update_wraplength(event=None):
    new_width = description_frame.winfo_width() - 40
    if new_width > 100:
        summary_label.configure(wraplength=new_width)

resize_after_id = None

def update_wraplength_wait(event=None):
    global resize_after_id
    if resize_after_id is not None:
        app.after_cancel(resize_after_id)
    resize_after_id = app.after(150, update_wraplength)

app.bind("<Configure>", update_wraplength_wait)

# Run the application
app.mainloop()