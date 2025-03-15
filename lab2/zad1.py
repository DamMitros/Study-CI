import pandas as pd, difflib

def correct_species_name(name,correct_species):
    name = name.strip().capitalize()
    if name in correct_species:
        return name
    else:
        correct_name = difflib.get_close_matches(name, correct_species, n=1, cutoff=0.6)
        if correct_name:
            return correct_name[0]
        else:
            print(f'Cannot find correct species for {name}')
            return name
    
df = pd.read_csv('iris_with_errors.csv')
# a)
empty_data = df.isna().sum() + (df == ' ').sum() + (df == '-').sum()
print(empty_data)

# b):
numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

for column in numeric_columns:
    median = df[column].median()
    invalid_data = (df[column] <= 0) | (df[column] >= 15) | df[column].isna() | (df[column] == ' ') | (df[column] == '-')
    df.loc[invalid_data, column] = median

df.to_csv('iris_corrected.csv', index=False)
print("The data has been corrected and saved to iris_corrected.csv")

# c):
species = df['variety'].unique()
correct_species = ['Setosa', 'Versicolor', 'Virginica']

if len(species) == 3 and all(species == correct_species):
    print("All species are correct")
else:
    print("There are some errors in species column")
    print(f'There are {len(species)} species in the dataset')
    print(f'Current species in file: {species}')

    df['variety'] = df['variety'].apply(lambda x: correct_species_name(x, correct_species))
    print(f'Corrected species in file: {df["variety"].unique()}')
    df.to_csv('iris_corrected.csv', index=False)
    print("The data has been corrected and saved to iris_corrected.csv")