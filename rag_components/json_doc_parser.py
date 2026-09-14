import json
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_json_documents(file_path: str) -> List[Document]:
    path = Path(file_path)

    # If the file path does not exist as given, try finding it relative to project root or current directory
    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        # Try stripping leading '../' or './'
        cleaned_filename = Path(file_path).name
        candidates = [
            project_root / cleaned_filename,
            Path.cwd() / cleaned_filename,
            project_root / file_path,
            Path.cwd() / file_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break

    if not path.is_file():
        raise FileNotFoundError(f"Could not find scheme JSON file at: {file_path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Flatten schemes if the JSON is organized by departments
    schemes_list = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "schemes" in item:
                dept_name = item.get("department_name", "")
                dept_url = item.get("department_url", "")
                for scheme in item.get("schemes", []):
                    scheme_copy = dict(scheme)
                    if "department_name" not in scheme_copy:
                        scheme_copy["department_name"] = dept_name
                    if "department_url" not in scheme_copy:
                        scheme_copy["department_url"] = dept_url
                    schemes_list.append(scheme_copy)
            elif isinstance(item, dict):
                schemes_list.append(item)

    documents: List[Document] = []

    for scheme in schemes_list:
        details = scheme.get("details", {})
        dept_name = scheme.get("department_name") or details.get("Concerned Department", "")
        scheme_title = scheme.get("scheme_title", "") or details.get("Scheme Title/Name", "")
        scheme_url = scheme.get("scheme_url", "")

        uploaded_file = details.get("Uploaded File", "-")
        if isinstance(uploaded_file, dict):
            file_info = f"{uploaded_file.get('text', '')} ({uploaded_file.get('url', '')})"
        else:
            file_info = str(uploaded_file)

        content = f"""Scheme Title: {scheme_title}
Department: {dept_name}
Organisation: {details.get("Organisation Name", "")}
Sponsored By: {details.get("Sponsered By", "")}
Beneficiaries: {details.get("Beneficiaries", "")}
Type of Benefit: {details.get("Types of Benefits", "")}
Funding Pattern: {details.get("Funding Pattern", "")}
How To Avail: {details.get("How To avail", "")}
Description: {details.get("Description", "")}
Uploaded File: {file_info}""".strip()

        document = Document(
            page_content=content,
            metadata={
                "scheme_title": scheme_title,
                "scheme_url": scheme_url,
                "department": dept_name,
            },
        )
        documents.append(document)

    return documents