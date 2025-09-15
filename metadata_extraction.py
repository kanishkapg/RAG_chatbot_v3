import os
from groq import Groq
from config import GROQ_API_KEY
import logging
import json
from database import get_all_texts, get_texts_missing_metadata, update_metadata
from datetime import datetime, date

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metadata_from_text(text_content: str):
    """
    Extracts issued date and effective date from text using an LLM.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not configured.")
        return None, None

    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts metadata from text. Extract the 'issued date' and 'effective date' from the provided text. Return the dates in 'YYYY-MM-DD' format. If a date is not found, return null for that field. Respond with only a JSON object with keys 'issued_date' and 'effective_date'."
                },
                {
                    "role": "user",
                    "content": text_content,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "json_object"},
        )
        
        response_text = chat_completion.choices[0].message.content
        metadata = json.loads(response_text)
        
        issued_date = metadata.get("issued_date")
        effective_date = metadata.get("effective_date")

        def parse_date(d):
            if not d:
                return None
            if isinstance(d, (date,)):
                return d
            if isinstance(d, str):
                # Try strict ISO first
                fmts = [
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y/%m/%d",
                    "%d %b %Y",
                    "%d %B %Y",
                ]
                for fmt in fmts:
                    try:
                        return datetime.strptime(d.strip(), fmt).date()
                    except Exception:
                        continue
            return None

        issued_date = parse_date(issued_date)
        effective_date = parse_date(effective_date)

        return issued_date, effective_date

    except Exception as e:
        logger.error(f"Failed to extract metadata using LLM: {e}")
        return None, None

def process_and_store_metadata(skip_if_present: bool = True):
    """
    Fetch texts, extract metadata, and update records.

    Args:
        skip_if_present: If True, only process records missing metadata.
    """
    logger.info("Starting metadata extraction process...")

    if skip_if_present:
        records = get_texts_missing_metadata()
        if not records:
            logger.info("All records already have metadata. Skipping extraction.")
            return
        logger.info(f"Found {len(records)} records missing metadata. Processing only those.")
    else:
        records = get_all_texts()
        logger.info(f"Processing all {len(records)} records for metadata extraction.")

    for record_id, text_content in records:
        if not text_content:
            continue
        logger.info(f"Extracting metadata for record {record_id}...")
        issued_date, effective_date = extract_metadata_from_text(text_content)
        if issued_date or effective_date:
            if update_metadata(record_id, issued_date, effective_date):
                logger.info(
                    f"Updated metadata for record {record_id}: Issued Date='{issued_date}', Effective Date='{effective_date}'"
                )
            else:
                logger.error(
                    f"Failed to update DB for record {record_id}. Dates: issued={issued_date}, effective={effective_date}"
                )
        else:
            logger.warning(f"No metadata extracted for record {record_id}.")
    logger.info("Metadata extraction process finished.")

if __name__ == '__main__':
    process_and_store_metadata()
