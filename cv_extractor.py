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

# def extract_skills_ner(text):
#     doc = nlp(text)
#     skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
#     return skills

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

def main():
    FILE_PATH = "./data/ENGINEERING/10219099.pdf"
    resume_text = extract_text_from_pdf(FILE_PATH)
    cleaned_text = preprocess_text(resume_text)
    # ner_skills = extract_skills_ner(cleaned_text)
    keyword_skills = extract_skills_keywords(cleaned_text)
    extracted_skills = extract_skills_from_resume(cleaned_text)
    education_info = extract_education(cleaned_text)
    formatted_education_info = format_education_info(education_info)

    # print("Skills Extracted using NER:", ner_skills)
    print("Skills Extracted:", keyword_skills)
    # print("Skills Extracted using YAKE:", extracted_skills)
    print("Education Found: ", education_info)
    for edu in formatted_education_info:
        print("Degree:", edu["Degree"])
        print("Major:", edu["Major"])
        print("University:", edu["University"])


if __name__ == '__main__':
    main()