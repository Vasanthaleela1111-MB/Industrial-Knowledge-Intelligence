import os

from pathlib import Path

from datetime import datetime


class MetadataExtractor:

    @staticmethod
    def extract(file_path):

        path = Path(file_path)

        return {

            "file_name": path.name,

            "extension": path.suffix,

            "size_mb": round(
                os.path.getsize(file_path)/1024/1024,
                2
            ),

            "created": datetime.fromtimestamp(
                os.path.getctime(file_path)
            ).isoformat(),

            "modified": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()
        }