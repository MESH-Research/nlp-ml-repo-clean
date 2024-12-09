import os
import time
import logging
import requests
import fitz # Note: fitz is better than Apache Tika as it gives better results for text extraction from a table.
import pytesseract
from PIL import Image
from docx import Document
from doc2docx import convert
import pptx
import sys # Deal with large text field in the output csv
import csv

#Make sure large text field in csv can be processed, this is useful when I examine the output.csv
csv.field_size_limit(sys.maxsize)

#Set up logging and make it display in my console
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def download_file(url, local_filename, auth_headers, max_retries=5, backoff_factor=1):
    """
    Downloads a file from the specified URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The path where the file will be saved.
        auth_headers (dict): HTTP headers containing authentication information.
        max_retries (int, optional): Maximum number of retry attempts. 5 is the default value.
        backoff_factor (int, optional): Delay factor for retries. 1 is the default value.
    
    Returns:
        str: The path to the downloaded file, or None if the download fails.
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting to download {url} (Attempt {attempt + 1})")
            response = requests.get(url, headers=auth_headers, allow_redirects=True) # Send HTTP request to the specific url
            response.raise_for_status() # Check for HTTP errors

            with open(local_filename, 'wb') as file: # Open local_filename in write mode
                for chunk in response.iter_content(chunk_size=8192): # Reads the response body in chunks of 8 KB
                    file.write(chunk)
            
            logging.info(f"Successfully downloaded {local_filename}")
            return local_filename # Exits the function download_file

        except requests.exceptions.RequestException as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(backoff_factor * (2 ** attempt)) # Delay increases with each failed attempt (1, 2, 4, 8s)

    logging.error(f"Failed to download {url} after {max_retries} attempts")
    return None
 
def extract_file(local_file_path, file_name, investigation_csv=None):
    """
    Extract text from a file based on its extension.

    Args:
        local_file_path (str): The path to the file.
        file_name (str): The name of the file.
        investigation_csv (str, optional): Path to the CSV file for logging unsupported files.
    
    Returns:
        str: Extracted text for supported files, or None for unsupported files.
    """
    # Switched to dictionary{} instead of list[]
    extraction_methods = {
        '.pdf', extract_text_from_pdf,
        '.docx', extract_text_from_docx,
        '.doc', extract_text_from_doc,
        '.pptx', extract_text_from_pptx,
        '.txt', extract_text_from_txt
    }

    # Extract file extension
    file_extension = os.path.splitext(file_name)[1].lower()

    # Find the matching function for the file extension
    extraction_function = extraction_methods.get(file_extension)

    if extraction_function:
        try:
            logging.info(f"Extracting text from {file_name} using {extraction_function.__name__}")
            return extraction_function(local_file_path)
        except Exception as e:
            logging.error(f"Error extracting text from {file_name} at {local_file_path}: {e}")
            return None
    else:
        # Handle unsupported files
        logging.warning(f"Unsupported file type for {file_name}. Skipping this file.")

        # Instead of simply return None, log skipped file to investigation CSV
        if investigation_csv:
            try:
                with open(investigation_csv, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([file_name, file_extension, "Unsupported"])
                    logging.info(f"Logged skipped file {file_name} to {investigation_csv}.")
            except Exception as e:
                logging.error(f"Failed to log skipped file {file_name} to {investigation_csv}: {e}")

        return None
    
def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from a PDF file.
    Uses PyMuPDF/fitz for text-based PDFs and PyTesseract for image-based PDFs.

    Args:
        pdf_file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text, or None if it fails.
    """
    try:
        with fitz.open(pdf_file_path) as doc:
            text = ""
            total_pages = len(doc)
            for page_num in range(total_pages): # Iterates through each page
                if (page_num + 1) % 10 == 0 or page_num == total_pages - 1: # Log every 10 pages or the last page
                    logging.info(f"Processing page {page_num + 1} of {total_pages}")

                page = doc.load_page(page_num) # Load the current page into memory

                try:
                    page_text = page.get_text()
                    if page_text.strip(): #If there is text on the page, it's likely text-based
                        logging.debug(f"Page {page_num + 1}: Text-based")
                        text += page_text
                    else: #This is probably image-based
                        logging.debug(f"Page {page_num + 1}: Image-based")
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB",[pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)

                        # Validate OCR results
                        if len(ocr_text.strip()) < 10:
                            logging.warning(f"OCR results on page {page_num + 1} may be incomplete.")
                            ocr_text = f"[Potential OCR issue on page {page_num + 1}]"
                        
                        text += ocr_text
                    
                    text += f"\n--- End of Page {page_num + 1} ---\n"
                except Exception as page_error:
                    logging.error(f"Error processing page {page_num +1}: {page_error}")
                    text += f"\n[Error extracting page {page_num + 1}]\n"

            return text

    except Exception as e:
        logging.error(f"Error processing PDF {pdf_file_path}: {e}")
        return None
    
def extract_text_from_docx(docx_file_path):
    """
    Extract text from a DOCX file.

    Args:
        docx_file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text, or None if it fails.
    """
    try:
        logging.info(f"Opening DOCX file: {docx_file_path}")
        doc = Document(docx_file_path)
        paragraphs = []

        for paragraph_num, paragraph in enumerate(doc.paragraphs, start=1):
            # Check if the paragraph text contains unexpected formatting or artifacts
            if paragraph.text.strip():
                clean_text = paragraph.text.strip()
                if "<" in clean_text or ">" in clean_text:
                    logging.warning(f"Potential formatting instructions detected in paragraph {paragraph_num}: {clean_text}")
                paragraphs.append(clean_text)
            else:
                logging.debug(f"Paragraph {paragraph_num} is empty and was skipped.")

        extracted_text = "\n".join(paragraphs)
        logging.info(f"Successfully extracted text from {len(paragraphs)} paragraphs.")
        return extracted_text

    except Exception as e:
        logging.error(f"Error extracting text from DOCX file {docx_file_path}: {e}")
        return None

def convert_doc_to_docx(doc_file_path):
    """
    Converts a .doc file to .docx format using the doc2docx library.

    Args:
        doc_file_path (str): Path to the .doc file to be converted.

    Returns:
        str: Path to the converted .docx file, or None if conversion fails.
    """
    try:
        # Define the output file name
        base_file_name = os.path.splitext(os.path.basename(doc_file_path))[0]
        newdocx_file_path = os.path.join(os.path.dirname(doc_file_path), f"{base_file_name}.docx")

        # Convert the file using doc2docx
        logging.info(f"Converting {doc_file_path} to {newdocx_file_path}")
        convert(doc_file_path, newdocx_file_path)

        if os.path.exists(newdocx_file_path):
            logging.info(f"Successfully converted {doc_file_path} to {newdocx_file_path}")
            return newdocx_file_path
        else:
            logging.error(f"Conversion failed. Converted file not found at {newdocx_file_path}")
            return None

    except Exception as e:
        logging.error(f"Error converting {doc_file_path} to .docx: {e}")
        return None

def extract_text_from_doc(doc_file_path):
    """
    Extract text from a .doc file. It will be converted to .docx before extraction.

    Args:
        doc_file_path (str): Path to the .doc file.

    Returns:
        str: Extracted text, or None if extraction fails.
    """
    try:
        if doc_file_path.endswith('.doc'):
            doc_file_path = convert_doc_to_docx(doc_file_path)
            if not doc_file_path:
                logging.error(f"Conversion of {doc_file_path} failed.")
                return None

        doc = Document(doc_file_path)
        paragraphs = []

        for para_num, para in enumerate(doc.paragraphs, start=1):
            if para.text.strip():
                paragraphs.append(para.text.strip())
            else:
                logging.debug(f"Skipped empty paragraph {para_num}.")
            
            extracted_text = "\n".join(paragraphs)
            logging.info(f"Successfully extracted {len(paragraphs)} paragraphs from {doc_file_path}")
            return extracted_text

    except Exception as e:
        logging.error(f"Error extracting text from {doc_file_path}: {e}")
        return None

def extract_text_from_txt(txt_file_path):
    """
    Extract text from a plain txt file using Python's built-in open() function.

    Args:
        txt_file_path (str): The path to the txt file.
    
    Returns:
        str: The extracted text, or None if it fails.
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        if not text.strip():
            logging.warning(f"Warning: The file {txt_file_path} is empty.")

        return text
    except Exception as e:
        logging.error(f"Error extracting text from TXT file: {e}")
        return None

def extract_text_from_pptx(pptx_file_path):
    """
    Extract text from a PowerPoint file, including slide text, speaker notes, and tables.

    Args:
        pptx_file_path (str): The path to the PowerPoint file.

    Returns:
        str: The extracted text, or None if it fails.
    """
    try:
        logging.info(f"Opening PowerPoint file: {pptx_file_path}")
        presentation = pptx.Presentation(pptx_file_path)
        all_slides_text = []

        for slide_num, slide in enumerate (presentation.slides, start=1):
            slide_text = []

            for shape in slide.shapes:
                # Extract text from text-containing shapes
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
                
                # Extract text from table
                if shape.shape_type == 19: # Type 19 matches to a table
                    for row in shape.table.rows:
                        for cell in row.cells:
                            cell_text = cell.text.strip()
                            if cell_text:
                                slide_text.append(cell_text)
            
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_text.append(f"Speaker Notes: {notes}")

            logging.info(f"Processed slide {slide_num}/{len(presentation.slides)}")
            
            if slide_text:
                # Combien slide content with headers
                all_slides_text.append(f"--- Slide {slide_num} ---\n" + "\n".join(slide_text))
            else:
                logging.warning(f"Slide {slide_num} contains no extractable text.")
            
        return "\n\n".join(all_slides_text)
    
    except Exception as e:
        logging.error(f"Error extracting text from PowerPoint file {pptx_file_path}: {e}")
        return None

def process_files(api_url, endpoint, api_key, output_csv, download_path):
    """
    Access files from an API, download them, extract text, and log results.
    
    Args:
        api_url (str): Base URL for the API.
        endpoint (str): API endpoint for fetching records.
        api_key (str): My personal authentication key for API requests, please switch to yours.
        output_csv (str): Path to the output CSV.
        download_path (str): Directory to save downloaded files.
    """
    try:
        auth_headers = {'Authorization': f'Bearer {api_key}'}
        page = 1
        page_size = 100 # Invenio limit
        has_more_pages = True

        # Updated headers to include 'DOI' and 'Flag' in the main CSV
        output_headers = ['Record ID', 'DOI', 'Languages', 'File Name', 'Extracted Text', 'Flag']

        # Make sure output directories and CSV exists.
        os.makedirs(download_path, exist_ok=True)

        if not os.path.exists(output_csv):
            with open(output_csv, 'w', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow(output_headers)

        while has_more_pages:
            response = requests.get(f"{api_url}/{endpoint}?size={page_size}&page={page}", headers=auth_headers)
            response.raise_for_status() # Raise an error for non-200 responses
            data = response.json()

            #This is where I check each page and all its records
            for record in data.get('hits', {}).get('hits', []):
                record_id = record['id']
                doi = record.get('pids', {}).get('doi', {}).get('identifier', 'N/A')  # Extract Doi if available
                languages = ','.join([lang['id'] for lang in record['metadata'].get('languages', [])])
                
                logging.info(f"Processing record {record_id} (DOI: {doi}, Languages: {languages})")

                for file_name, file_metadat in record.get('files', {}).get('entries', {}).items():
                    file_url = f"{api_url}/api/records/{record_id}/files/{file_name}/content"
                    local_file_path = os.path.join(download_path, file_name)

                    # Download the file
                    download_result = download_file(file_url, local_file_path, auth_headers)
                    if download_result:
                        extracted_text = extract_file(local_file_path, file_name)
                        flag = 0 if extracted_text else 1  # Flag: success (0) or failure (1)

                        # Append results to the output CSV
                        with open(output_csv, 'a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerow([record_id, doi, languages, file_name, extracted_text or '[Error: Extraction Failed]', flag])

                        # Start removing downloaded file if processed successfully
                        if flag == 0:
                                os.remove(local_file_path)
                        else:
                            logging.warning(f"Failed to process file {file_name} for record {record_id}. Logged for investigation.")
                    else:
                        logging.error(f"Failed to download file {file_name} for record {record_id}.")

            # Check if there are more pages to fetch
            logging.info(f"Completed page {page}.")
            has_more_pages = 'next' in data.get('links', {})
            page += 1

        logging.info("All files processed and results written to CSV.")

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def main():
    """Main function that leads the workflow"""
    api_url = "https://works.hcommons.org/"
    api_key = "waiting for an updated API key"
    api_endpoint = "api/records"
    output_csv = "output9.csv"
    download_path = "download_files9"

    os.makedirs(download_path, exist_ok=True)
    process_files(api_url, api_endpoint, api_key, output_csv, download_path)

if __name__ == "__main__":
    main()