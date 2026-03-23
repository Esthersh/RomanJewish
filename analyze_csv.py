import pandas as pd

df = pd.read_csv("/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv")
print(f"Total rows: {len(df)}")

# Check 'ref Code' column types in the WHOLE CSV
print("\n'ref Code' values in whole CSV (first 20):")
print(df['ref Code'].head(20))

coerced_ref_code = pd.to_numeric(df['ref Code'], errors='coerce')
nan_count_ref_code = coerced_ref_code.isna().sum()
print(f"\nNon-numeric 'ref Code' count in whole CSV: {nan_count_ref_code} out of {len(df)}")

# Check 'SourceID' column
print("\n'SourceID' values in whole CSV (first 20):")
print(df['SourceID'].head(20))
coerced_source_id = pd.to_numeric(df['SourceID'], errors='coerce')
nan_count_source_id = coerced_source_id.isna().sum()
print(f"\nNon-numeric 'SourceID' count in whole CSV: {nan_count_source_id} out of {len(df)}")

# Filter records with Keywords and English
filtered = df.dropna(subset=["Keywords", "English"])
print(f"\nRows with Keywords and English: {len(filtered)}")

# Try to convert to numeric in filtered set
coerced_filtered = pd.to_numeric(filtered['Refference'], errors='coerce')
nan_count_filtered = coerced_filtered.isna().sum()
print(f"Non-numeric Refference count in filtered set: {nan_count_filtered}")

if nan_count > 0:
    print("Example non-numeric references:")
    print(filtered[coerced.isna()]['Refference'].head(5))
    print("Do they have keywords?")
    print(filtered[coerced.isna()][['Keywords', 'English']].head(5))
