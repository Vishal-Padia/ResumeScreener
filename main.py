import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load job descriptions from "job_descriptions.csv"
job_descriptions_df = pd.read_csv("job_descriptions.csv")
job_descriptions = job_descriptions_df["Job Description"].tolist()

# Load merged data from "merged.csv"
merged_df = pd.read_csv("merged.csv")
if "Top Skills" in merged_df.columns:
    resume_skills = merged_df["Top Skills"].fillna('').str.split(',').apply(lambda x: [s.strip() for s in x]).tolist()
elif "Skills" in merged_df.columns:
    resume_skills = merged_df["Skills"].fillna('').str.split(',').apply(lambda x: [s.strip() for s in x]).tolist()
else:
    raise ValueError("No 'Top Skills' or 'Skills' column found in merged.csv")

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to preprocess and get embeddings for text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()  # Convert to numpy array

# Function to calculate cosine similarity between two sets of embeddings
def calculate_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(np.array(embeddings1), np.array(embeddings2))

# Create a DataFrame to store results
results_df = pd.DataFrame(columns=["Job Description", "Selected Resumes", "Similarity Scores"])

# Iterate through job descriptions
for job_description in tqdm(job_descriptions, desc="Processing Job Descriptions"):
    job_description_embeddings = get_embeddings(job_description)

    top_resumes = []
    top_similarity_scores = []

    # Calculate cosine similarity scores for each resume individually
    for resume_skill in resume_skills:
        resume_embeddings = get_embeddings(' '.join(resume_skill))
        similarity_score = calculate_cosine_similarity(job_description_embeddings, resume_embeddings)
        top_resumes.append(resume_skill)  # Append the skill as a representation of the resume
        top_similarity_scores.append(similarity_score[0][0])

    # Find the top 3 resumes with highest similarity scores
    top_indices = np.argsort(top_similarity_scores)[-3:][::-1]
    top_resumes = [top_resumes[i] for i in top_indices]
    top_similarity_scores = [top_similarity_scores[i] for i in top_indices]

    new_row = {
        "Job Description": job_description,
        "Selected Resumes": ', '.join([' '.join(resume) for resume in top_resumes]),
        "Similarity Scores": ', '.join(map(str, top_similarity_scores))
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Save results to a CSV file
results_df.to_csv("job_description_matching_results.csv", index=False)
