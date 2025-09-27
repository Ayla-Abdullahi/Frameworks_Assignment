import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# -----------------------------
# Load data safely
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        engine='python',       # safer for large/malformed CSVs
        on_bad_lines='skip',   # skip rows with errors
        encoding='utf-8'       # proper handling of special characters
    )
    # Convert publish_time to datetime, extract year
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    # Add abstract word count
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    # Drop rows missing essential info
    df_clean = df.dropna(subset=['title', 'publish_time'])
    return df_clean

# -----------------------------
# File path
# -----------------------------
file_path = "../data/metadata_sample.csv"
df = load_data(file_path)

# -----------------------------
# Streamlit layout
# -----------------------------
st.title("🧠 CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers")

# -----------------------------
# Data Overview
# -----------------------------
st.subheader("📊 Dataset Overview")
st.write("Basic information about the dataset:")
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
st.write("Column types:")
st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']))
st.write("Missing values per column:")
st.dataframe(df.isnull().sum())
st.write("Basic statistics for numerical columns:")
st.dataframe(df.describe())

# -----------------------------
# Sidebar: Year filter
# -----------------------------
year_range = st.slider(
    "Select publication year range",
    int(df['year'].min()),
    int(df['year'].max()),
    (2020, 2021)
)

df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
st.write(f"Showing {df_filtered.shape[0]} papers between {year_range[0]} and {year_range[1]}.")

# -----------------------------
# Visualization: Publications by Year
# -----------------------------
st.subheader("📈 Publications by Year")
year_counts = df_filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax, color="skyblue")
ax.set_title("Publications by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
st.pyplot(fig)

# -----------------------------
# Visualization: Top Journals
# -----------------------------
st.subheader("🏛️ Top Journals")
top_journals = df_filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax, color="lightcoral")
ax.set_title("Top 10 Journals")
ax.set_xlabel("Number of Papers")
ax.set_ylabel("Journal")
st.pyplot(fig)

# -----------------------------
# Visualization: Papers by Source
# -----------------------------
st.subheader("📰 Papers by Source")
if 'source_x' in df_filtered.columns:
    source_counts = df_filtered['source_x'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(y=source_counts.index, x=source_counts.values, ax=ax, color="mediumseagreen")
    ax.set_title("Top Sources")
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("Source")
    st.pyplot(fig)
else:
    st.write("Column 'source_x' not found in dataset.")
    # the papers by Source wont show as the sample metadata is too shalow for it.

# -----------------------------
# Word Cloud of Titles
# -----------------------------
st.subheader("💬 Word Cloud of Titles")
titles = " ".join(df_filtered['title'].dropna().tolist())
wc = WordCloud(width=800, height=400, background_color="white").generate(titles)
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# -----------------------------
# Top Words Table (optional)
# -----------------------------
st.subheader("🔠 Top Words in Titles")
all_words = " ".join(df_filtered['title'].dropna().tolist()).lower().split()
common_words = Counter(all_words).most_common(20)
st.table(pd.DataFrame(common_words, columns=['Word', 'Count']))

# -----------------------------
# Sample of Data
# -----------------------------
st.subheader("📝 Sample Data")
st.dataframe(df_filtered[['title','authors','journal','year']].head(20))
