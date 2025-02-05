from datasets import load_dataset

# Load the dataset
dataset = load_dataset("simplescaling/s1K_tokenized")

# Access the dataset split (usually 'train' for single split datasets)
data = dataset['train']

# Convert to pandas for analysis if needed
df = data.to_pandas()

# Print basic info
print(df.info())
print("\nFirst few rows:")
print(df.head())
