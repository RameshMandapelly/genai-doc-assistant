import json
from pathlib import Path

import pandas as pd
import yaml
from PyPDF2 import PdfReader


def extract_text(file_path: str) -> str:
    """
    Dispatches to the right extractor based on file extension.
    Returns the extracted content as a single text string.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".txt":
        return _extract_txt(path)
    elif extension == ".pdf":
        return _extract_pdf(path)
    elif extension in (".csv", ".xlsx", ".xls"):
        return _extract_tabular(path)
    elif extension == ".json":
        return _extract_json(path)
    elif extension in (".yaml", ".yml"):
        return _extract_yaml(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)


def _extract_tabular(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # Convert each row to a readable text line
    rows_as_text = df.astype(str).apply(lambda row: " | ".join(row), axis=1)
    return "\n".join(rows_as_text)


def _extract_json(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def _extract_yaml(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return yaml.dump(data, default_flow_style=False)