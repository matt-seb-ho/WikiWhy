import os
import pandas as pd

file_column_mapping = {
    "context.json": ['ctx', 'title', 'topic', 'split'], 
    "question.json": ['question'],
    "cause.json": ['cause'],
    "effect.json": ['effect'],
    "explanation.json": ['explanation']
}

# Load the WikiWhy dataset into a Pandas DataFrame
def load_wikiwhy(directory_path):
    df = pd.DataFrame()
    for filename, column_subset in file_column_mapping.items():
        sub_df = pd.read_json(os.path.join(directory_path, filename))
        df[column_subset] = sub_df[column_subset]
    return df
