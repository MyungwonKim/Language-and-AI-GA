import pandas as pd
import re
import html
import ftfy
from langdetect import detect, LangDetectException
from tqdm import tqdm # Progress bar

# --- CONFIGURATION ---
INPUT_FILE = r"C:/Users/20235050/Downloads/BDS_Y3/Language_AI/assignment_data/assignment_data/extrovert_introvert.csv"
OUTPUT_FILE = r"C:/Users/20235050/Downloads/BDS_Y3/Language_AI/assignment_data/processed_reddit_authors_full.csv" # Renamed slightly to indicate full text
AUTHOR_COL = "auhtor_ID"   
POST_COL = "post"       
LABEL_COL = "extrovert" 

# --- PREPROCESSOR CLASS ---
class RedditPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.user_pattern = re.compile(r'u/\S+')
        self.sub_pattern = re.compile(r'r/\S+')
        
        # Regex for quotes (handles spaces before >)
        self.quote_pattern = re.compile(r'^\s*>.*$', re.MULTILINE)
        
        self.symbol_squash_pattern = re.compile(r'([!?.@$])\1{2,}')
        self.markdown_link_pattern = re.compile(r'\[(.*?)\]\(.*?\)')
        
        self.bot_phrases = [
            "i am a bot", "action was performed automatically", 
            "submission has been removed", "contact the moderators"
        ]

    def clean_post(self, text):
        if not isinstance(text, str): return ""
        
        # 1. Fix Encoding & HTML
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        
        # 2. Remove Quotes & Markdown Links
        text = self.quote_pattern.sub('', text)
        text = self.markdown_link_pattern.sub(r'\1', text)
        
        # 3. Bot Check
        if any(phrase in text.lower() for phrase in self.bot_phrases):
            return ""

        # 4. Token Replacements
        text = self.url_pattern.sub('[URL]', text)
        text = self.user_pattern.sub('[USER]', text)
        text = self.sub_pattern.sub('[SUB]', text)
        
        # 5. Symbol Squashing & Whitespace
        text = self.symbol_squash_pattern.sub(r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Language Check
        if len(text) < 5: 
            return text if text.isascii() else ""
            
        try:
            if detect(text) != 'en': return ""
        except LangDetectException:
            return ""

        return text

# --- MAIN EXECUTION ---
def process_data():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='latin1')
        
    print(f"Original shape: {df.shape}")
    
    # 1. Initialize Preprocessor
    processor = RedditPreprocessor()
    tqdm.pandas(desc="Cleaning Posts")
    
    # 2. Apply Cleaning
    print("Cleaning posts...")
    df['clean_text'] = df[POST_COL].progress_apply(processor.clean_post)
    
    # 3. Remove Empty Rows
    df = df[df['clean_text'] != ""]
    
    # 4. Group by Author and Aggregate
    print("Grouping by Author...")
    agg_funcs = {'clean_text': lambda x: ' '.join(x)}
    if LABEL_COL in df.columns:
        agg_funcs[LABEL_COL] = 'first'
        
    df_grouped = df.groupby(AUTHOR_COL).agg(agg_funcs).reset_index()
    
    # 5. NO TRUNCATION (Changed Step)
    # We directly copy the full history to final_text.
    # This keeps 100% of the data for your chunking strategy later.
    print("Copying full text (No Truncation)...")
    df_grouped['final_text'] = df_grouped['clean_text']
    
    # 6. Save
    # We drop clean_text to avoid having two massive columns, 
    # but final_text will still be very large.
    cols_to_keep = [AUTHOR_COL, 'final_text']
    if LABEL_COL in df_grouped.columns:
        cols_to_keep.append(LABEL_COL)
    
    df_final = df_grouped[cols_to_keep]
    
    print(f"Final grouped shape: {df_final.shape}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
