import customtkinter as ctk
from tkinter import filedialog
import fitz
import webbrowser

#setting UI RELATED STUFF
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("Resume + Job Matcher")
app.geometry("900x600")

#save resume path(uploaded resume)
resume_path = ctk.StringVar()
#the job the user clicked on(to view detail)
current_selected_title = None
#access the job by its title(here we assume no duplicate job titles, change this if we have any -> very unlikely though)
job_card_refs = {}

#check if the input includes only digits
def only_digits(input):
    return input.isdigit() or input == ""


#happens when user clicks on "upload resume(pdf)"
def upload_resume():
    path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if path:
        resume_path.set(path)

#use fitz to read all text in pdf(no pictures, they aren't typically in resume anyway)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

#happens when we click "navigate, basically bring the user to the "
def open_job_url(url):
    webbrowser.open_new(url)

#happens when the user click on one of the searched jobs
def toggle_description(title, summary, salary, location, years, card_frame):
    global current_selected_title

    #if we already selected a title, unselect it(do not show its description)
    if current_selected_title == title:
        current_selected_title = None
        description_frame.grid_remove()
        card_frame.configure(fg_color="#f1f1f1")
        return

    #otherwise we show the selected job info
    current_selected_title = title
    for card in job_card_refs.values():
        card.configure(fg_color="#f1f1f1")#white for unselected
    card_frame.configure(fg_color="#dbeafe")#light blue for selected

    #populate the description based on input(aka based on job info)
    job_title_label.configure(text=title)
    salary_label.configure(text=f"💵 Salary: {salary or 'N/A'}")
    location_label.configure(text=f"📍 Location: {location or 'N/A'}")
    experience_label.configure(text=f"📆 Required Experience: {years or 'N/A'} years")
    summary_label.configure(text=summary)

    #make a grid to place these things
    description_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))

#create job in ui(happens after I click submit and get all sort of job information)
def add_job(title, url, summary, salary, location, years):
    #create the box for this job and tie it with the specific title
    job_card = ctk.CTkFrame(job_list_frame, fg_color="#f1f1f1", cursor="hand2")
    job_card.pack(fill="x", padx=5, pady=5)
    job_card_refs[title] = job_card

    #second layer, for the padding effect
    card_body = ctk.CTkFrame(job_card, fg_color="transparent")
    card_body.pack(fill="x", padx=5, pady=5)

    #title and button of this job
    title_label = ctk.CTkLabel(card_body, text=title, font=("Segoe UI", 12, "bold"), anchor="w", width=280)
    title_label.pack(side="left")

    #if i click on the button, i should be navigated to the correlated url
    ctk.CTkButton(card_body, text="Navigate →", width=80, command=lambda: open_job_url(url)).pack(side="right")

    # basically if i click anywhere besides the button, i should activate the description of this job
    for i in [job_card, card_body, title_label]:
        i.bind("<Button-1>", lambda e: toggle_description(title, summary, salary, location, years, job_card))

#well, happens when I click submit
#TODO: make sure the salary min and max are numerical values(hint the users that this is in usd)
#same for experience, just numerical value, must be year
def submit():
    #clear the jobs in the list(as we are loading new ones)
    global job_card_refs
    job_card_refs = {}

    #destroy the correlated job ui items too
    for i in job_list_frame.winfo_children():
        i.destroy()
    
    #nothing is selected, obviously
    global current_selected_title
    current_selected_title = None

    #there shouldnt be any description too, as the jobs are brand new(and are unselected)
    description_frame.grid_remove()

    #remove this in the future, for debug purpose only(to see what's read and what's not)
    min_salary = salary_min.get()
    max_salary = salary_max.get()
    years_exp = experience.get()
    desired_loc = location.get()

    print("\n=== User Input ===")
    print("Salary Min:", min_salary)
    print("Salary Max:", max_salary)
    print("Experience:", years_exp)
    print("Location:", desired_loc)

    #print resume info too
    resume = resume_path.get()
    if resume:
        print("\n=== Resume Text ===")
        text = extract_text_from_pdf(resume)
        print(text)
    else:
        print("\n(No resume uploaded)")





    #create two demo jobs(for testing purpose only)
    add_job("Software Engineer at Google", "https://careers.google.com/jobs/",
            "Work on scalable systems, AI, and web technologies. Collaborate across teams on services impacting billions.",
            "$120k–$180k", "Mountain View, CA", "3+")

    add_job("AI Intern at OpenAI", "https://openai.com/careers/",
            "Contribute to AGI research, tools, and deployment. Work with world-class researchers on real-world impact.",
            "$80/hr", "San Francisco, CA", "1+")

#UI beautification, thanks to the sample UI framework created by AI
#my original version look way too ugly

#basically left frame will pack user input and entry
#with some buttons and entry boxes
left_frame = ctk.CTkFrame(app, width=250)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

validate_num = app.register(only_digits)

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

ctk.CTkButton(left_frame, text="Upload Resume (PDF)", command=upload_resume).pack(pady=10)
ctk.CTkLabel(left_frame, textvariable=resume_path, wraplength=250).pack()
ctk.CTkButton(left_frame, text="Submit", command=submit).pack(pady=15)

#the right side, include both the place to select jobs and the description part(do a grid of 1 column and 2 rows)
main_right_frame = ctk.CTkFrame(app)
main_right_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
main_right_frame.grid_rowconfigure(0, weight=3)#job listing will take 60% of screen if description exists, else 100%(height)
main_right_frame.grid_rowconfigure(1, weight=2)#weight 3:2
main_right_frame.grid_columnconfigure(0, weight=1)

#job list on the top
job_list_frame_container = ctk.CTkFrame(main_right_frame)
job_list_frame_container.grid(row=0, column=0, sticky="nsew")

job_list_canvas = ctk.CTkScrollableFrame(job_list_frame_container, width=550)
job_list_canvas.pack(fill="both", expand=True, padx=10)
job_list_frame = job_list_canvas

#description on the bottom(remove as it is not selected in the initial state)
#with some string as its content
description_frame = ctk.CTkFrame(main_right_frame, fg_color="#ffffff", corner_radius=8)
description_frame.grid_remove()

job_title_label = ctk.CTkLabel(description_frame, text="", font=("Arial", 16, "bold"), anchor="w")
job_title_label.pack(anchor="w", padx=10, pady=(5, 2))

salary_label = ctk.CTkLabel(description_frame, text="", anchor="w")
salary_label.pack(anchor="w", padx=10)

location_label = ctk.CTkLabel(description_frame, text="", anchor="w")
location_label.pack(anchor="w", padx=10)

experience_label = ctk.CTkLabel(description_frame, text="", anchor="w")
experience_label.pack(anchor="w", padx=10)

summary_label = ctk.CTkLabel(description_frame, text="", anchor="w", wraplength=700, justify="left")
summary_label.pack(anchor="w", padx=10, pady=(10, 5))

#just run this
app.mainloop()
