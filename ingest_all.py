"""Send an ingest event for every supported document in a folder (recursively)."""
import argparse
import asyncio
import os
from pathlib import Path

import inngest

from config import SUPPORTED_EXTENSIONS
from main import inngest_client


async def main(folder: str) -> None:
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    print(f"Found {len(files)} document(s) in {folder}")

    for path in files:
        source_id = path.name
        await inngest_client.send(
            inngest.Event(
                name="rag/ingest_document",
                data={"path": str(path.resolve()), "source_id": source_id},
            )
        )
        print(f"Sent ingest event for: {path} (source_id={source_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=os.getenv("DOCS_FOLDER", os.getenv("PDF_FOLDER", "./pdfs")))
    asyncio.run(main(parser.parse_args().folder))
