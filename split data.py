import pandas as pd
from sklearn.model_selection import train_test_split

# Load csv file containing dataset
csv_path = "hasil_baca_file.csv"
df = pd.read_csv(csv_path)

# Print the first few rows of the dataframe to check
print("First few rows of the dataframe:")
print(df.head())

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print the first few rows of the training set
print("\nFirst few rows of the training set:")
print(train_df.head())

# Print the first few rows of the testing set
print("\nFirst few rows of the testing set:")
print(test_df.head())

# Optionally, save the training and testing sets to new CSV files
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
