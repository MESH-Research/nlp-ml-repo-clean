import pandas as pd
import os
import sys
import csv

# Ensure large CSV fields can be processed
csv.field_size_limit(sys.maxsize)

# cd path/to/your/folder
# Run this in terminal `python examine_output9.py`

def examine_csv(file_path, output_report):
    """
    Analyze a CSV file and generate a report with basic statistics, language analysis, text length analysis, and problemtic rows.

    Args:
        file_path(str): path to the csv file to examine.
        output_report(str): path to the report in .md.
    """
    try:
        # Load the csv
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)

        # Basic Overview
        print(f"Analyzing basic statistics.")
        num_rows = len(df)
        column_info = df.dtypes.to_dict()
        missing_values = df.isnull().sum().to_dict()

        # Content Inspection
        print(f"Inspecting content.")
        df['Text Length'] = df['Extracted Text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        avg_length = df['Text Length'].mean()
        median_length = df['Text Length'].median()
        min_length = df ['Text Length'].min()
        max_length = df['Text Length'].max()

        # Language Analysis
        print(f"Analayzing language.")
        language_counts = df['Languages'].value_counts().to_dict()

        # Flag Analysis
        print(f"Analyzing flags.")
        flag_counts = df['Flag'].value_counts().to_dict() if 'Flag' in df.columns else {}

        # Problematic Rows
        print("Identifying problemtic rows.")
        empty_text_rows = df[df['Extracted Text'].isnull()]
        empty_language_rows = df[df['Languages'].isnull()]

        # Write to Report
        print(f"Writing report to {output_report}")
        with open(output_report, 'w') as report:
            report.write("# CSV Analysis Report\n")
            report.write(f"##File Analyzed:** {file_path}\n\n")
            report.write("## Basic Statistics\n")
            report.write(f"- Total Rows: {num_rows}\n")
            report.write(f"- Column Info: {column_info}\n")
            report.write(f"- Missing Values: {missing_values}\n\n")
            
            report.write("## Text Length Analysis\n")
            report.write(f"- Average Length: {avg_length:.2f}\n") # 2 means round the number to 2 decimal places; f means fixed point notation
            report.write(f"- Median Length: {median_length}\n")
            report.write(f"- Minimum Length: {min_length}\n")
            report.write(f"- Maximum Length: {max_length}\n\n")

            report.write("## Language Analysis\n")
            report.write("Language Counts:\n")
            for lang, count in language_counts.items():
                report.write(f"- {lang}: {count}\n")
            report.write("\n")

            report.write("## Flag Analysis\n")
            report.write("Flag Counts:\n")
            for flag, count in flag_counts.items():
                report.write(f"- {flag}: {count}\n")
            report.write("\n")

            report.write("## Problematic Rows\n")
            report.write(f"Empty Text Rows: {len(empty_text_rows)}\n")
            report.write(f"Empty Language Rows: {len(empty_language_rows)}\n\n")
            if len(empty_text_rows) > 0:
                report.write("### Rows with Empty Text\n")
                report.write(empty_text_rows.to_string(index=False))
                report.write("\n\n")
            if len(empty_language_rows) > 0:
                report.write("### Rows with Empty Language\n")
                report.write(empty_language_rows.to_string(index=False))
                report.write("\n")

        print("Report generated successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    csv_file_path = "/Users/TianyiKou/Documents/GitHub/nlp-ml-repo-clean/output9.csv" # Your path
    report_output_path = "output9_analysis_report.md"
    examine_csv(csv_file_path, report_output_path)

if __name__ == "__main__":
    main()
