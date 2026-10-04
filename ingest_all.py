
import asyncio
import os
from pathlib import Path

import inngest
from main import inngest_client

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")


async def main():
    pdf_files = sorted(Path(PDF_FOLDER).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {PDF_FOLDER}")

    for pdf_path in pdf_files:
        source_id = pdf_path.stem
        await inngest_client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "pdf_path": str(pdf_path),
                    "source_id": source_id,
                },
            )
        )
        print(f"Sent ingest event for: {pdf_path.name} (source_id={source_id})")


if __name__ == "__main__":
    asyncio.run(main())