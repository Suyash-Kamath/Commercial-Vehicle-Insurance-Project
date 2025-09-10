# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# from io import BytesIO
# import base64
# import json
# import unicodedata
# import os
# import requests
# from dotenv import load_dotenv
# import logging

# # Check if openai package is available
# try:
#     from openai import OpenAI
# except ImportError as e:
#     logging.error("OpenAI package not found. Please install it using 'pip install openai'")
#     raise ImportError("OpenAI package not found. Please install it using 'pip install openai'")

# load_dotenv()
# app = FastAPI()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://127.0.0.1:5500"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load OpenAI API key from environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     logger.error("OPENAI_API_KEY environment variable not set")
#     raise ValueError("OPENAI_API_KEY environment variable not set")

# # Initialize OpenAI client
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except Exception as e:
#     logger.error(f"Failed to initialize OpenAI client: {str(e)}")
#     raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

# @app.post("/process-file/")
# async def process_file(file: UploadFile = File(...), company_name: str = Form("Unknown Company")):
#     try:
#         logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")
        
#         # Check if file is empty
#         if file.size == 0:
#             logger.error(f"File {file.filename} is empty")
#             raise HTTPException(status_code=400, detail=f"File {file.filename} is empty. Please upload a valid file.")
        
#         # Read file content
#         file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
#         file_type = file.content_type if file.content_type else file_extension
#         logger.info(f"Detected file extension: {file_extension}, content_type: {file_type}")
        
#         allowed_extensions = ['txt', 'csv', 'xlsx', 'xls']
#         allowed_mime_types = [
#             'text/plain',
#             'text/csv',
#             'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # xlsx
#             'application/vnd.ms-excel'  # xls
#         ]

#         if file_extension not in allowed_extensions and file_type not in allowed_mime_types:
#             logger.error(f"Unsupported file: {file.filename}, extension: {file_extension}, content_type: {file_type}")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported file type for {file.filename}. Extension ({file_extension}) or MIME type ({file_type}) not allowed. Only .txt, .csv, .xlsx, and .xls are supported."
#             )
        
#         if file_extension in ['txt'] or file_type in ['text/plain']:
#             extracted_text = (await file.read()).decode('utf-8', errors='ignore')
#             extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
#             logger.info("Text file processed successfully")
        
#         elif file_extension == 'csv' or file_type == 'text/csv':
#             try:
#                 df = pd.read_csv(file.file, encoding='utf-8', errors='ignore')
#                 extracted_text = df.to_string()
#                 extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
#                 logger.info("CSV file processed successfully")
#             except Exception as e:
#                 logger.error(f"Error reading CSV file {file.filename}: {str(e)}")
#                 raise HTTPException(status_code=400, detail=f"Error reading CSV file {file.filename}: {str(e)}")
        
#         elif file_extension in ['xlsx', 'xls'] or file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
#             try:
#                 all_sheets = pd.read_excel(file.file, sheet_name=None)
#                 dfs = []
#                 for sheet_name, df_sheet in all_sheets.items():
#                     df_sheet["Source_Sheet"] = sheet_name
#                     dfs.append(df_sheet)
#                 df = pd.concat(dfs, ignore_index=True, join="outer")

#                 extracted_text = df.to_string(index=False, justify="left", max_cols=None, max_rows=None)
#                 extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
#                 logger.info("Excel file (all sheets, all columns) processed successfully")
#             except Exception as e:
#                 logger.error(f"Error reading Excel file {file.filename}: {str(e)}")
#                 raise HTTPException(status_code=400, detail=f"Error reading Excel file {file.filename}: {str(e)}")
        
#         logger.info("Sending text to OpenAI for parsing")
#         # Parse extracted text
#         parse_prompt = f"""
#         Analyze the following text, which contains policy or insurance details (e.g., segments, payouts (PO), types, locations, transactions, ages, districts, etc.).
        
#         Extract and structure into these fields:
#         - Segment: e.g., PCV 3Wheeler Electric
#         - Location: Combine RTO State and District info, e.g., Assam - All Districts
#         - Policy Type: e.g., Comp.+TP All
#         - Payout: e.g., 60%
#         - Remarks: Only list applicable conditions (e.g., Old transactions, All ages). Do not include exclusions.
        
#         Output in JSON format:
#         {{
#             "segment": "Segment value",
#             "location": "Location value",
#             "policy_type": "Policy Type value",
#             "payout": "Payout value",
#             "remarks": "Applicable remarks, comma-separated"
#         }}
        
#         Text: {extracted_text.replace('{', '{{').replace('}', '}}')}
#         """
        
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a precise extractor of insurance policy data. Be accurate and only include applicable items in remarks."},
#                     {"role": "user", "content": parse_prompt.encode('ascii', 'ignore').decode('ascii')}
#                 ],
#                 temperature=0.1
#             )
#             parsed_json = response.choices[0].message.content.strip()
#             logger.info("OpenAI parsing successful")
#         except OpenAI.OpenAIError as oai_e:
#             logger.error(f"OpenAI API error: {str(oai_e)}")
#             raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(oai_e)}. Check your API key or rate limits.")
#         except requests.exceptions.ConnectionError as conn_e:
#             logger.error(f"Network connection error: {str(conn_e)}")
#             raise HTTPException(status_code=503, detail="Network connection error. Please check your internet connection.")
        
#         if parsed_json.startswith('```json'):
#             parsed_json = parsed_json[7:-3].strip()
        
#         try:
#             data = json.loads(parsed_json)
#             logger.info("JSON parsing successful")
#         except json.JSONDecodeError as json_e:
#             logger.error(f"JSON decode error: {str(json_e)}, Raw response: {parsed_json}")
#             raise HTTPException(status_code=500, detail=f"Failed to parse JSON response: {str(json_e)}. Raw response: {parsed_json}")
        
#         # Create Excel output
#         df_data = pd.DataFrame({
#             'Segment': [data.get('segment', '')],
#             'Location': [data.get('location', '')],
#             'Policy Type': [data.get('policy_type', '')],
#             'Payout': [data.get('payout', '')],
#             'Remarks': [data.get('remarks', '')]
#         })
        
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df_data.to_excel(writer, sheet_name='Policy Data', startrow=1, index=False)
#             workbook = writer.book
#             worksheet = writer.sheets['Policy Data']
#             headers = ['Segment', 'Location', 'Policy Type', 'Payout', 'Remarks']
#             for col_num, value in enumerate(headers, 1):
#                 worksheet.cell(row=2, column=col_num, value=value)
#             company_cell = worksheet.cell(row=1, column=1, value=company_name)
#             worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=5)
#             company_cell.font = company_cell.font.copy(bold=True, size=14)
        
#         output.seek(0)
#         logger.info("Excel file generated successfully")
        
#         # Return JSON with extracted text, parsed data, and Excel file as base64
#         excel_base64 = base64.b64encode(output.read()).decode('utf-8')
#         return JSONResponse(content={
#             "extracted_text": extracted_text,
#             "parsed_data": data,
#             "excel_file": excel_base64
#         })
    
#     except Exception as e:
#         logger.error(f"Unexpected error for file {file.filename}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred for file {file.filename}: {str(e)}")


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import base64
import json
import unicodedata
import os
import requests
from dotenv import load_dotenv
import logging

# Check if openai package is available
try:
    from openai import OpenAI
except ImportError as e:
    logging.error("OpenAI package not found. Please install it using 'pip install openai'")
    raise ImportError("OpenAI package not found. Please install it using 'pip install openai'")

load_dotenv()
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), company_name: str = Form("Unknown Company")):
    try:
        logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")
        
        if file.size == 0:
            logger.error(f"File {file.filename} is empty")
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty. Please upload a valid file.")
        
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        file_type = file.content_type if file.content_type else file_extension
        logger.info(f"Detected file extension: {file_extension}, content_type: {file_type}")
        
        allowed_extensions = ['txt', 'csv', 'xlsx', 'xls']
        allowed_mime_types = [
            'text/plain',
            'text/csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]

        if file_extension not in allowed_extensions and file_type not in allowed_mime_types:
            logger.error(f"Unsupported file: {file.filename}, extension: {file_extension}, content_type: {file_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {file.filename}. Only .txt, .csv, .xlsx, and .xls are supported."
            )
        
        if file_extension in ['txt'] or file_type in ['text/plain']:
            extracted_text = (await file.read()).decode('utf-8', errors='ignore')
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        elif file_extension == 'csv' or file_type == 'text/csv':
            df = pd.read_csv(file.file, encoding='utf-8', errors='ignore')
            extracted_text = df.to_string()
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        elif file_extension in ['xlsx', 'xls'] or file_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]:
            all_sheets = pd.read_excel(file.file, sheet_name=None)
            dfs = []
            for sheet_name, df_sheet in all_sheets.items():
                df_sheet["Source_Sheet"] = sheet_name
                dfs.append(df_sheet)
            df = pd.concat(dfs, ignore_index=True, join="outer")

            extracted_text = df.to_string(index=False, justify="left", max_cols=None, max_rows=None)
            extracted_text = unicodedata.normalize('NFKD', extracted_text).encode('ascii', 'ignore').decode('ascii')
        
        # ---- Send to OpenAI ----
        parse_prompt = f"""
        Analyze the following text, which contains policy or insurance details (e.g., segments, payouts, types, locations, etc.).
        Extract and structure into these fields:
        - Segment
        - Location
        - Policy Type
        - Payout
        - Remarks

        Output in JSON format:
        {{
            "segment": "Segment value(s), comma-separated if multiple",
            "location": "Location value(s), comma-separated if multiple",
            "policy_type": "Policy Type value(s), comma-separated if multiple",
            "payout": "Payout value(s), comma-separated if multiple",
            "remarks": "Remark(s), comma-separated if multiple"
        }}

        Text: {extracted_text.replace('{', '{{').replace('}', '}}')}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise extractor of insurance policy data."},
                {"role": "user", "content": parse_prompt.encode('ascii', 'ignore').decode('ascii')}
            ],
            temperature=0.1
        )
        parsed_json = response.choices[0].message.content.strip()
        if parsed_json.startswith('```json'):
            parsed_json = parsed_json[7:-3].strip()
        
        data = json.loads(parsed_json)

        # ---- Create Excel output (separate rows for multiple values) ----
        segments = [s.strip() for s in data.get('segment', '').split(',') if s.strip()]
        locations = [l.strip() for l in data.get('location', '').split(',') if l.strip()]
        policy_types = [p.strip() for p in data.get('policy_type', '').split(',') if p.strip()]
        payouts = [po.strip() for po in data.get('payout', '').split(',') if po.strip()]
        remarks_list = [r.strip() for r in data.get('remarks', '').split(',') if r.strip()]

        if not segments: segments = [""]
        if not locations: locations = [""]
        if not policy_types: policy_types = [""]
        if not payouts: payouts = [""]
        if not remarks_list: remarks_list = [""]

        rows = []
        for seg in segments:
            for loc in locations:
                for pol in policy_types:
                    for pay in payouts:
                        for rem in remarks_list:
                            rows.append({
                                "Segment": seg,
                                "Location": loc,
                                "Policy Type": pol,
                                "Payout": pay,
                                "Remarks": rem
                            })

        df_data = pd.DataFrame(rows)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_data.to_excel(writer, sheet_name='Policy Data', startrow=1, index=False)
            workbook = writer.book
            worksheet = writer.sheets['Policy Data']
            headers = ['Segment', 'Location', 'Policy Type', 'Payout', 'Remarks']
            for col_num, value in enumerate(headers, 1):
                worksheet.cell(row=2, column=col_num, value=value)
            company_cell = worksheet.cell(row=1, column=1, value=company_name)
            worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=5)
            company_cell.font = company_cell.font.copy(bold=True, size=14)

        output.seek(0)

        excel_base64 = base64.b64encode(output.read()).decode('utf-8')
        return JSONResponse(content={
            "extracted_text": extracted_text,
            "parsed_data": data,
            "excel_file": excel_base64
        })
    
    except Exception as e:
        logger.error(f"Unexpected error for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred for file {file.filename}: {str(e)}")
