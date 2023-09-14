import pandas as pd
from datasets import load_dataset

def save_job_descriptions_to_csv(output_file, num_descriptions):
    try:
        # Load the job descriptions dataset from Hugging Face
        dataset = load_dataset("jacob-hugging-face/job-descriptions")

        # Extract the specified number of job descriptions
        job_data = dataset["train"][:num_descriptions]

        # Create a DataFrame
        df = pd.DataFrame({
            "Company Name": job_data["company_name"],
            "Job Description": job_data["job_description"],
            "Position Title": job_data["position_title"]
        })

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)

        print(f"{num_descriptions} job descriptions saved to {output_file}")
    except Exception as e:
        print("Error:", str(e))

def main():
    save_job_descriptions_to_csv(output_file="job_descriptions.csv", num_descriptions=50)

if __name__ == '__main__':
    main()
