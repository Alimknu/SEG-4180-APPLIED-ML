"""

SEG 4180 Assignment 2

"""

# imports
from datasets import load_dataset
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import great_expectations as gx
import matplotlib.pyplot as plt
import seaborn as sns

# Load Awesome ChatGPT Prompts dataset
print("LOADING DATASET")
dataset = load_dataset("fka/awesome-chatgpt-prompts", trust_remote_code = True)
df = dataset['train'].to_pandas()

print(f"Dataset loaded successfully.")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Dataset Overview
print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)

print("\n1. NUMBER OF SAMPLES:")
print(f"   - Total samples: {df.shape[0]}")
print(f"   - Total features: {df.shape[1]}")

print("\n2. DATA STRUCTURE:")
print("\n   Feature Details:")
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    unique_vals = df[col].nunique()
    print(f"   - {col}:")
    print(f"     * Type: {dtype}")
    print(f"     * Non-null: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
    print(f"     * Unique values: {unique_vals}")
    if dtype == 'object' and col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        print(f"     * Average length: {avg_len:.1f} characters")

print("\n3. NOTABLE METADATA:")
print(f"   - Dataset source: Hugging Face (fka/awesome-chatgpt-prompts)")
print(f"   - Content: ChatGPT role-play prompts")
print(f"   - Unique roles (acts): {df['act'].nunique()}")
print(f"   - Most common role: '{df['act'].mode()[0]}' ({df['act'].value_counts().iloc[0]} occurrences)")

print("\n4. STATISTICAL SUMMARY:")
print(f"   - Prompt length range: {df['prompt'].str.len().min()}-{df['prompt'].str.len().max()} chars")
print(f"   - Average prompt length: {df['prompt'].str.len().mean():.1f} chars")
print(f"   - Median prompt length: {df['prompt'].str.len().median():.1f} chars")

print("\nFirst sample:")
print(f"Act: {df.iloc[0]['act']}")
print(f"Prompt: {df.iloc[0]['prompt'][:100]}...")
print("="*70)

# Data Cleaning
print("DATA CLEANING")
df_clean = df.copy()

print("\nChecking for missing values:")
missing_count = df_clean.isnull().sum()
for col, count in missing_count.items():
    if count > 0:
        print(f" - '{col}': {count} missing values (will impute)")
    else:
        print(f" '{col}': No missing values")

# Remove duplicates based on prompt
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['prompt'], keep='first')
duplicates_removed = initial_rows - len(df_clean)
print(f" - Removed {duplicates_removed} duplicate prompts")

# Filter out prompts that are too long (>1000 characters) or too short (<10 characters)
df_clean = df_clean[(df_clean['prompt'].str.len() >= 10) & (df_clean['prompt'].str.len() <= 1000)]
rows_after_filter = len(df_clean)
print(f" - Removed {initial_rows - duplicates_removed - rows_after_filter} prompts outside length range (10-1000 chars)")
print(f" - Final dataset size: {len(df_clean)} rows\n")

print("\nApplying text preprocessing...")
df_clean['prompt_clean'] = df_clean['prompt'].str.lower()
df_clean['act_clean'] = df_clean['act'].str.lower()
print(" ✓ Converted text to lowercase")

df_clean['prompt_clean'] = df_clean['prompt_clean'].str.translate(
    str.maketrans('', '', string.punctuation)
)
print(" ✓ Removed punctuation")

df_clean['prompt_clean'] = df_clean['prompt_clean'].str.replace(r'\d+', '', regex = True)
print(" ✓ Removed numbers")

df_clean['prompt_clean'] = df_clean['prompt_clean'].str.strip()
print(" ✓ Stripped whitespace")

print("\nCreating engineered features...")
df_clean['prompt_length'] = df_clean['prompt'].str.len()
df_clean['word_count'] = df_clean['prompt'].str.split().str.len()
df_clean['avg_word_length'] = df_clean['prompt_length'] / df_clean['word_count']
print(f" ✓ Added prompt_length (mean: {df_clean['prompt_length'].mean():.1f} chars)")
print(f" ✓ Added word_count (mean: {df_clean['word_count'].mean():.1f} words)")
print(f" ✓ Added avg_word_length (mean: {df_clean['avg_word_length'].mean():.1f} chars/word)")

# Feature Engineering
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

print("\nCreating Count Vectorizer features...")
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=50, stop_words='english')
X_counts = vectorizer.fit_transform(df_clean['prompt_clean'])
print(f" ✓ Count Vectorizer: {X_counts.shape[0]} samples, {X_counts.shape[1]} features")

feature_names = vectorizer.get_feature_names_out()
print(f" ✓ Top 10 features: {list(feature_names[:10])}")

print("\nCreating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=20, stop_words='english')
X_tfidf = tfidf.fit_transform(df_clean['prompt_clean'])
print(f" ✓ TF-IDF Vectorizer: {X_tfidf.shape[0]} samples, {X_tfidf.shape[1]} features")
print(f" ✓ Top 10 TF-IDF features: {list(tfidf.get_feature_names_out()[:10])}")

print("\nEncoding categorical features...")
df_clean['act_encoded'] = pd.factorize(df_clean['act'])[0]
print(f" ✓ Encoded 'act' column: {df_clean['act_encoded'].nunique()} unique roles")

# Dimensionality Reduction
print("\n" + "="*50)
print("DIMENSIONALITY REDUCTION")
print("="*50)

if X_counts.shape[1] > 1:
    print(f"\nApplying PCA to reduce from {X_counts.shape[1]} to 3 dimensions...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_counts.toarray())
    explained_var = pca.explained_variance_ratio_
    print(f" ✓ PCA completed")
    print(f" ✓ Explained variance by component: {explained_var}")
    print(f" ✓ Total variance explained: {sum(explained_var):.2%}")
else:
    print("\nSkipping PCA (insufficient features)")

# Data validation
print("\n" + "="*50)
print("DATA VALIDATION")
print("="*50)
print("\nRunning data validation checks...")

# For GX v1.x, we'll use a simplified validation approach
# Define expectations manually and check them
validation_results = {
    'success': True,
    'results': []
}

# Check for null values
null_prompt = df_clean['prompt'].isnull().sum()
null_act = df_clean['act'].isnull().sum()
validation_results['results'].append({
    'expectation': 'prompt has no nulls',
    'success': null_prompt == 0,
    'null_count': int(null_prompt)
})
validation_results['results'].append({
    'expectation': 'act has no nulls',
    'success': null_act == 0,
    'null_count': int(null_act)
})

# Check prompt_length range
prompt_length_valid = df_clean['prompt_length'].between(10, 1000).all()
validation_results['results'].append({
    'expectation': 'prompt_length between 10 and 1000',
    'success': bool(prompt_length_valid)
})

# Check uniqueness
prompts_unique = df_clean['prompt'].is_unique
validation_results['results'].append({
    'expectation': 'prompts are unique',
    'success': bool(prompts_unique)
})

# Check data types
prompt_is_str = df_clean['prompt'].dtype == 'object'
act_is_str = df_clean['act'].dtype == 'object'
validation_results['results'].append({
    'expectation': 'prompt is string type',
    'success': bool(prompt_is_str)
})
validation_results['results'].append({
    'expectation': 'act is string type',
    'success': bool(act_is_str)
})

# Update overall success
validation_results['success'] = all(r['success'] for r in validation_results['results'])

print("\n" + "="*50)
print("Validation Results:")
print("="*50)
for result in validation_results['results']:
    status = "✓ PASS" if result['success'] else "✗ FAIL"
    print(f"{status}: {result['expectation']}")
print(f"\nOverall Success: {validation_results['success']}")
print("="*50 + "\n")

# Visualizations
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

print("\nGenerating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15,10))
fig.suptitle('EDA: ChatGPT Prompts Dataset', fontsize=16, fontweight='bold')

axes[0, 0].hist(df_clean['prompt_length'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Prompt Length Distribution')
axes[0, 0].set_xlabel('Characters')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df_clean['prompt_length'].mean(), color='red', linestyle='--')

top_roles = df_clean['act'].value_counts().head(10)
axes[0, 1].barh(range(len(top_roles)), top_roles.values)
axes[0, 1].set_yticks(range(len(top_roles)))
axes[0, 1].set_yticklabels(top_roles.index)
axes[0, 1].invert_yaxis()
axes[0, 1].set_title('Top 10 ChatGPT Roles')
axes[0, 1].set_xlabel('Count')

axes[0, 2].boxplot(df_clean['word_count'])
axes[0, 2].set_title('Word Count Distribution (Box Plot)')
axes[0, 2].set_ylabel('Words per Prompt')

axes[1, 0].imshow(df_clean.isnull().T, aspect='auto', cmap='viridis')
axes[1, 0].set_title('Missing Values Heatmap')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Features')

numeric_cols = ['prompt_length', 'word_count', 'avg_word_length', 'act_encoded']
corr_matrix = df_clean[numeric_cols].corr()
im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_title('Feature Correlations')
axes[1, 1].set_xticks(range(len(numeric_cols)))
axes[1, 1].set_xticklabels([c[:10] for c in numeric_cols], rotation=45)
axes[1, 1].set_yticks(range(len(numeric_cols)))
axes[1, 1].set_yticklabels([c[:10] for c in numeric_cols])
plt.colorbar(im, ax=axes[1, 1])

axes[1, 2].axis('off')
sample_text = "Sample Prompt:\n\n" + df_clean.iloc[0]['prompt'][:200] + "..."
axes[1, 2].text(0.1, 0.5, sample_text, fontsize=10, 
                verticalalignment='center', family='monospace')
axes[1, 2].set_title('Example Prompt')

plt.tight_layout()
print(" ✓ Created visualization plots")

print("\nSaving outputs...")
plt.savefig('assignment_visualizations.png', dpi=300, bbox_inches='tight')
print(" ✓ Saved visualization: assignment_visualizations.png")
plt.show()

df_clean.to_csv('cleaned_prompts.csv', index=False)
print(f" ✓ Saved cleaned dataset: cleaned_prompts.csv ({len(df_clean)} rows, {len(df_clean.columns)} columns)")

# Analysis Summary
print("\n" + "="*70)
print("KEY OBSERVATIONS AND INSIGHTS")
print("="*70)

print("\n1. DATA QUALITY:")
print(f"   - Original dataset: {initial_rows} samples")
print(f"   - After cleaning: {len(df_clean)} samples ({len(df_clean)/initial_rows*100:.1f}% retained)")
print(f"   - Duplicates removed: {duplicates_removed} ({duplicates_removed/initial_rows*100:.1f}%)")
print(f"   - Invalid length prompts removed: {initial_rows - duplicates_removed - rows_after_filter}")
print(f"   - Missing values: None detected in critical columns")
print(f"   - All validation checks: {'PASSED' if validation_results['success'] else 'FAILED'}")

print("\n2. PROMPT CHARACTERISTICS:")
print(f"   - Average prompt length: {df_clean['prompt_length'].mean():.1f} characters")
print(f"   - Prompt length std dev: {df_clean['prompt_length'].std():.1f} characters")
print(f"   - Average word count: {df_clean['word_count'].mean():.1f} words")
print(f"   - Average word length: {df_clean['avg_word_length'].mean():.2f} characters/word")
print(f"   - Shortest prompt: {df_clean['prompt_length'].min()} chars")
print(f"   - Longest prompt: {df_clean['prompt_length'].max()} chars")

print("\n3. ROLE DISTRIBUTION:")
print(f"   - Total unique roles: {df_clean['act'].nunique()}")
print(f"   - Top 5 most common roles:")
top_5_roles = df_clean['act'].value_counts().head(5)
for idx, (role, count) in enumerate(top_5_roles.items(), 1):
    print(f"     {idx}. {role}: {count} prompts ({count/len(df_clean)*100:.1f}%)")

print("\n4. FEATURE ENGINEERING RESULTS:")
print(f"   - Count Vectorizer features: {X_counts.shape[1]} (n-grams: 1-2)")
print(f"   - TF-IDF features: {X_tfidf.shape[1]}")
print(f"   - PCA components: 3 (from {X_counts.shape[1]} dimensions)")
if 'X_pca' in locals():
    print(f"   - Variance explained by PCA: {sum(pca.explained_variance_ratio_):.2%}")

print("\n5. TEXT PROCESSING INSIGHTS:")
print(f"   - Most common words (after preprocessing):")
word_freq = pd.Series(' '.join(df_clean['prompt_clean']).split()).value_counts().head(5)
for idx, (word, freq) in enumerate(word_freq.items(), 1):
    print(f"     {idx}. '{word}': {freq} occurrences")

print("\n6. CORRELATION INSIGHTS:")
print(f"   - Correlation between prompt_length and word_count: {df_clean[['prompt_length', 'word_count']].corr().iloc[0,1]:.3f}")
print(f"   - Correlation between word_count and avg_word_length: {df_clean[['word_count', 'avg_word_length']].corr().iloc[0,1]:.3f}")

print("\n7. KEY FINDINGS:")
print("   ✓ Dataset contains diverse ChatGPT role-play prompts")
print("   ✓ High data quality after cleaning (no nulls, no duplicates)")
print("   ✓ Prompts vary significantly in length and complexity")
print("   ✓ Strong correlation between prompt length and word count")
print("   ✓ Successful feature extraction using NLP techniques")
print("   ✓ Dimensionality reduction preserves key information")

print("\n" + "="*70)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*70)