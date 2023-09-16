# Resume Matching with Job Descriptions Using PDF CVs

## Overview
This project aims to automate the process of matching job descriptions with candidate resumes in PDF format. The primary goal is to efficiently extract relevant details from CVs and compare them with job descriptions to find the most suitable candidates for specific roles.

## Objective
The main objective of this project is to create a robust system that streamlines the resume screening process, making it faster and more accurate. By automating the matching of job descriptions with candidate resumes, organizations can identify potential candidates more efficiently, ultimately saving time and resources.

## Approach
### 1. PDF Data Extraction
- **Dataset:** We start by acquiring a dataset of resumes in PDF format from Kaggle, known as the "Kaggle Resume Dataset."
- **PDF Extraction:** We build a PDF extractor using Python libraries like PyPDF2 or PDFMiner to extract essential information from the CVs. Key details include the candidate's job role category, skills, and education (degree and institution).

### 2. Job Description Data Understanding
- **Dataset:** We fetch job descriptions from the Hugging Face dataset, focusing on obtaining a diverse set of 10-15 job descriptions.
- **Comprehension:** To ensure effective matching, we comprehensively analyze and understand the job descriptions, including the required skills and qualifications.

### 3. Candidate-Job Matching
- **Tools:** We utilize the Transformers library by Hugging Face, with models like BERT or DistilBERT as the foundation for embedding extraction.
- **Tokenization and Preprocessing:** Both job descriptions and extracted CV details are tokenized and preprocessed to prepare them for embedding.
- **Embedding Conversion:** The tokenized text is converted into embeddings using pretrained models like DistilBERT from Hugging Face.
- **Cosine Similarity Calculation:** For each job description, we calculate the cosine similarity between its embedding and the embeddings of the CVs.
- **Ranking Candidates:** CVs are ranked based on their similarity scores for each job description.
- **Top Candidates:** We list the top 5 CVs for each job description, considering those with the highest similarity scores.

## Outcome
The outcome of this project is an efficient resume matching system that can significantly reduce the time and effort required for screening candidates. It provides organizations with a shortlist of the most relevant candidates for each job description, improving the overall hiring process.

## Note
This project encourages innovation and flexibility in tool selection. While we suggest specific tools and models, you are welcome to explore alternatives that you believe may enhance the project's performance. Please document any deviations from the recommendations and provide a rationale for your choices.
