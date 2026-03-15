# """
# app/services/document_loader.py

# Ingestion pipeline for property documents.
# Loads files, attaches metadata, and stores in the vector store.

# Called by:
#     scripts/ingest_documents.py  — CLI bulk ingestion
#     api/routes/documents.py      — HTTP upload endpoint

# Not called during request handling (chat, search).
# """
# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import Optional

# import structlog

# from app.services.rag_service import ingest_document as rag_ingest_document

# logger = structlog.get_logger(__name__)

# SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# DOC_TYPE_KEYWORDS = {
#     "lease":      "lease",
#     "contract":   "contract",
#     "strata":     "strata",
#     "inspection": "inspection_report",
#     "disclosure": "disclosure",
#     "title":      "title_deed",
#     "rates":      "council_rates",
#     "body_corp":  "body_corporate",
# }


# class DocumentLoaderService:

#     def ingest_file(
#         self,
#         file_path: str,
#         property_id: str,
#         doc_type: Optional[str] = None,
#         extra_metadata: Optional[dict] = None,
#     ) -> dict:
#         """
#         Ingest a single document into the vector store.

#         Returns:
#             {success, filename, property_id, doc_type, chunk_count}
#         """
#         path = Path(file_path).resolve()
#         self._validate_file(path)

#         filename = path.name
#         detected_doc_type = doc_type or self._detect_doc_type(filename)

#         metadata = {
#             "property_id": property_id,
#             "doc_type": detected_doc_type,
#             "filename": filename,
#             "file_path": str(path),
#             **(extra_metadata or {}),
#         }

#         logger.info(
#             "ingest_file",
#             property_id=property_id,
#             doc_type=detected_doc_type,
#             filename=filename,
#         )

#         chunk_count = rag_ingest_document(
#             file_path=str(path),
#             metadata=metadata,
#         )

#         logger.info("ingest_complete", filename=filename, chunks=chunk_count)

#         return {
#             "success": True,
#             "filename": filename,
#             "property_id": property_id,
#             "doc_type": detected_doc_type,
#             "chunk_count": chunk_count,
#         }

#     def ingest_directory(
#         self,
#         directory_path: str,
#         property_id: str,
#         recursive: bool = False,
#     ) -> list[dict]:
#         """
#         Ingest all supported documents in a directory.
#         Returns list of per-file results including any failures.
#         """
#         directory = Path(directory_path).resolve()

#         if not directory.is_dir():
#             raise DocumentLoaderError(f"Directory not found: {directory}")

#         pattern = "**/*" if recursive else "*"
#         files = [
#             f for f in directory.glob(pattern)
#             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
#         ]

#         if not files:
#             logger.warning("no_documents_found", directory=str(directory))
#             return []

#         logger.info("ingest_directory", file_count=len(
#             files), directory=str(directory))

#         results = []
#         for file_path in files:
#             try:
#                 result = self.ingest_file(
#                     file_path=str(file_path),
#                     property_id=property_id,
#                 )
#                 results.append(result)
#             except Exception as e:
#                 logger.error("ingest_file_failed",
#                              filename=file_path.name, error=str(e))
#                 results.append({
#                     "success": False,
#                     "filename": file_path.name,
#                     "property_id": property_id,
#                     "error": str(e),
#                 })

#         success_count = sum(1 for r in results if r.get("success"))
#         logger.info("ingest_directory_complete",
#                     success=success_count, total=len(results))

#         return results

#     def _validate_file(self, path: Path):
#         if not path.exists():
#             raise DocumentLoaderError(f"File not found: {path}")
#         if not path.is_file():
#             raise DocumentLoaderError(f"Not a file: {path}")
#         if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
#             raise DocumentLoaderError(
#                 f"Unsupported type: {path.suffix}. "
#                 f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
#             )
#         if path.stat().st_size == 0:
#             raise DocumentLoaderError(f"File is empty: {path.name}")

#     def _detect_doc_type(self, filename: str) -> str:
#         filename_lower = filename.lower()
#         for keyword, doc_type in DOC_TYPE_KEYWORDS.items():
#             if keyword in filename_lower:
#                 return doc_type
#         return "document"


# class DocumentLoaderError(Exception):
#     pass
