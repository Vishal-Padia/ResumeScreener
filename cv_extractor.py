import os
import csv
from PyPDF2 import PdfReader
import spacy
import re
import yake

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file_path):
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print("Error:", str(e))
        return None

def preprocess_text(text):
    # Clean and preprocess the text (lowercase, remove punctuation)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text 

def extract_skills_keywords(text):
    # Create a YAKE keyword extractor
    keyword_extractor = yake.KeywordExtractor()

    # Extract keywords from the preprocessed text
    skill_keywords_extracted = keyword_extractor.extract_keywords(text)

    # Extract the keywords from the tuples
    skill_keywords = [keyword for keyword, score in skill_keywords_extracted]

    return skill_keywords

def extract_skills_from_resume(resume_text):
    # Preprocess the text
    preprocessed_text = preprocess_text(resume_text)

    # Extract skills using keyword extraction
    extracted_skills = extract_skills_keywords(preprocessed_text)

    return extracted_skills

def extract_education(text):
    education_info = []

    # Define regular expression for education details (major and university)
    education_pattern = re.compile(
                        r"(?i)(?P<Degree>Bachelor|B\.S\.|Master|M\.S\.|Ph\.D\.|Doctorate).*?"
                        r"(?P<Major>[\w\s]+).*?"
                        r"(?P<University>[\w\s,]+)"
    )

    # Search for education details using the pattern
    matches = education_pattern.findall(text)

    for match in matches:
        degree, major, university = match
        education_info.append({
            "Degree": degree.strip(),
            "Major": major.strip(),
            "University": university.strip()
        })

    return education_info

def format_education_info(education_info):
    formatted_info = []
    for edu in education_info:
        formatted_info.append({
            "Degree": edu["Degree"],
            "Major": edu["Major"],
            "University": edu["University"],
        })
    return formatted_info

def extract_category_from_folder(folder_path):
    return os.path.basename(folder_path)

def extract_top_skills_from_folder(folder_path, top_n):
    top_skills_per_cv = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                cv_path = os.path.join(root, file)
                resume_text = extract_text_from_pdf(cv_path)
                cleaned_text = preprocess_text(resume_text)
                top_skills = extract_skills_from_resume(cleaned_text)
                top_skills_per_cv.append(top_skills[:top_n])
    return top_skills_per_cv

def extract_education_and_category(folder_path):
    education_and_category_info = []
    category_name = extract_category_from_folder(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                cv_path = os.path.join(root, file)
                resume_text = extract_text_from_pdf(cv_path)
                cleaned_text = preprocess_text(resume_text)
                education_info = extract_education(cleaned_text)
                education_and_category_info.append({
                    "CV File": cv_path,
                    "Category": category_name,
                    "Education": education_info,
                })
    return education_and_category_info

def save_cv_data_to_csv(category_folder, output_csv_folder, top_n=10):
    top_skills_per_cv = extract_top_skills_from_folder(category_folder, top_n)
    education_and_category_info = extract_education_and_category(category_folder)
    category_name = extract_category_from_folder(category_folder)
    output_csv_path = os.path.join(output_csv_folder, f"{category_name}.csv")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["CV File", "Category", "Education", "Top Skills"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in education_and_category_info:
            cv_path = item["CV File"]
            category = item["Category"]
            education = item["Education"]
            top_skills = top_skills_per_cv.pop(0)
            writer.writerow({
                "CV File": cv_path,
                "Category": category,
                "Education": education,
                "Top Skills": ', '.join(top_skills),
            })

def main():
    DATA_FOLDER = "./data"  # Replace with your folder path
    OUTPUT_CSV_FOLDER = "./output_csv"
    top_n = 10

    if not os.path.exists(OUTPUT_CSV_FOLDER):
        os.makedirs(OUTPUT_CSV_FOLDER)

    for root, dirs, _ in os.walk(DATA_FOLDER):
        for category in dirs:
            category_folder = os.path.join(DATA_FOLDER, category)
            save_cv_data_to_csv(category_folder, OUTPUT_CSV_FOLDER, top_n)

    print(f"CSV files saved to {OUTPUT_CSV_FOLDER}")

if __name__ == '__main__':
    main()
