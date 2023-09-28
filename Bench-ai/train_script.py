from benchkit.data.helpers import get_dataloader
from benchkit.tracking.config import Config
import pandas as pd
import torch 
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from functools import partial

"""
- Follow this tutorial for help writing your training script: https://docs.bench-ai.com/Tutorials/train-script
- Once you are ready to train your model run `python manage.py migrate-code <Version#>`
- Then follow this tutorial to start training sessions: https://docs.bench-ai.com/Tutorials/Experiments
"""

tracker_config = {
    "name":"resumescreener"
} # The hyperparameters you wish to keep track of

model_config = Config(tracker_config, # hyperparams we are using for this model
                      "dummy_value", # We will be evaluating this model based on validation loss
                      "max") # We will be trying to minimize validation loss

# Define a directory to save checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Function to save checkpoint
def save_checkpoint(results_df, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.csv")
    results_df.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
    return checkpoint_dir


# GPU for Bench-AI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Ensures Cuda is used
print(device)   

# Load job descriptions from "job_descriptions.csv"
job_descriptions_df = pd.read_csv("datasets/job_descriptions.csv")
job_descriptions = job_descriptions_df["Job Description"].tolist()

# Load merged data from "merged.csv"    
merged_df = pd.read_csv("datasets/merged.csv")
if "Top Skills" in merged_df.columns:
    resume_skills = merged_df["Top Skills"].fillna('').str.split(',').apply(lambda x: [s.strip() for s in x]).tolist()
elif "Skills" in merged_df.columns:
    resume_skills = merged_df["Skills"].fillna('').str.split(',').apply(lambda x: [s.strip() for s in x]).tolist()
else:
    raise ValueError("No 'Top Skills' or 'Skills' column found in merged.csv")

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = model.to(device) 

# Function to preprocess and get embeddings for text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()  # Convert to numpy array

# Function to calculate cosine similarity between two sets of embeddings
def calculate_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(np.array(embeddings1), np.array(embeddings2))

def main():
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

        # Find the top 3 resumes with the highest similarity scores
        top_indices = np.argsort(top_similarity_scores)[-3:][::-1]
        top_resumes = [top_resumes[i] for i in top_indices]
        top_similarity_scores = [top_similarity_scores[i] for i in top_indices]

        new_row = {
            "Job Description": job_description,
            "Selected Resumes": ', '.join([' '.join(resume) for resume in top_resumes]),
            "Similarity Scores": ', '.join(map(str, top_similarity_scores))
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the final results to a CSV file
        final_results_path = "results/job_description_matching_results.csv"
        results_df.to_csv(final_results_path, index=False)
        print(f"Final results saved at: {final_results_path}")

        # Save checkpoint at regular intervals (every 2 job descriptions)
        if len(results_df) % 2 == 0:
            save_checkpoint(results_df, len(results_df) // 2)
            locked_function = partial(save_checkpoint, results_df, len(results_df) // 2) 
            model_config.save_model_and_state(locked_function,
                                      lambda : results_dir,
                                      len(results_df) // 2,
                                      float(len(results_df) // 2)) # Saves a checkpoint, and a Save if appropriate

if __name__ == '__main__':
    main()
