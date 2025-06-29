# find specific instances:

corpus_path="/Users/yaronot/Desktop/Data Courses/Plagiarism Detection/pan-plagiarism-corpus-2011-1/external-detection-corpus"
csv_file="all_plagiarism_mappings.csv"

# Extract a specific row (e.g., row 1, which is the first data row)
extract_row 1 "$csv_file" "$corpus_path"