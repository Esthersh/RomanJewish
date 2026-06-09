"""Helpers for loading the LUR corpus from the Google Drive-hosted .xlsx file.

The file is an uploaded Excel workbook (not a native Google Sheet), so we
download it via the Drive API and read it with pandas/openpyxl.

The canonical row identifier is the ``Id`` column (header name, not a column
letter) — use ``id_column()`` to get it consistently.
"""

import io
import tomllib
from pathlib import Path

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

PROJECT_ROOT = Path(__file__).parent.parent
SECRETS_FILE = PROJECT_ROOT / "src" / ".streamlit" / "secrets.toml"

DRIVE_FILE_ID = "1LhGuaQEos3YsLkEsL0evKRsDIg6iXVfr"
DEFAULT_TAB = "Samples2Update"
ID_COLUMN = "Id"


def _download_xlsx(file_id: str) -> io.BytesIO:
    with open(SECRETS_FILE, "rb") as f:
        secrets = tomllib.load(f)
    creds_info = secrets["connections"]["gsheets"]
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    drive = build("drive", "v3", credentials=creds)
    req = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf


def load_samples_tab(tab: str = DEFAULT_TAB, file_id: str = DRIVE_FILE_ID) -> pd.DataFrame:
    """Download the workbook and return the named tab as a string-typed DataFrame."""
    buf = _download_xlsx(file_id)
    df = pd.read_excel(buf, sheet_name=tab, dtype=str)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Tab {tab!r} is missing required {ID_COLUMN!r} column. Found: {list(df.columns)}")
    return df
