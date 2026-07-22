from pathlib import Path

from ingestion.constants import SUPPORTED_DOCUMENTS


class FileDetector:

    @staticmethod
    def detect(file_path: str):

        extension = Path(file_path).suffix.lower()

        return SUPPORTED_DOCUMENTS.get(
            extension,
            "Unsupported"
        )