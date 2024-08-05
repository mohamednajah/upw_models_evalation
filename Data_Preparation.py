import pandas as pd

input_file = 'Datensatz-Fachaufgabe-Data-to-Text-Manager.xlsx'
df = pd.read_excel(input_file)

print("Initial DataFrame Info:")
print(df.info())
print("\nInitial DataFrame Head:")
print(df.head())

df['Beschreibung'] = df['Beschreibung'].astype(str).str.strip()

df = df.dropna(subset=['Beschreibung'])
df = df[df['Beschreibung'] != '']


df['target'] = df['Variationsgruppe'].apply(lambda x: 1 if 'positive' in str(x).lower() else 0)

df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data has been saved to: cleaned_data.csv")

