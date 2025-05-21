"""Upload ingested Markdown files as a Hugging Face dataset."""

from __future__ import annotations

import glob
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import custom_save_dataset


@dataclass(slots=True)
class IngestedDocument:
    document_id: str
    document_text: str
    document_filename: str
    document_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UploadStageConfig:
    run: bool = False
    source_documents_dir: str | None = None


def _load_config(config: dict[str, Any]) -> UploadStageConfig:
    stage = config.get("pipeline", {}).get("upload_ingest_to_hub", {})
    return UploadStageConfig(
        run=bool(stage.get("run")), source_documents_dir=stage.get("source_documents_dir")
    )


def _resolve_source_dir(cfg: UploadStageConfig, config: dict[str, Any]) -> str:
    if cfg.source_documents_dir:
        return cfg.source_documents_dir
    ingestion_dir = config.get("pipeline", {}).get("ingestion", {}).get("output_dir")
    if ingestion_dir:
        logger.debug("source_documents_dir not provided; using ingestion output_dir")
        return ingestion_dir
    raise ValueError("No source directory configured for upload_ingest_to_hub stage")


def _read_markdown(path: str) -> IngestedDocument | None:
    try:
        text = open(path, "r", encoding="utf-8").read().strip()
    except Exception as exc:  # pragma: no cover - just in case
        logger.error(f"Failed reading {path}: {exc}")
        return None
    if not text:
        logger.warning(f"Skipping empty markdown file: {path}")
        return None
    doc_id = str(uuid.uuid4())
    logger.debug(f"Loaded markdown file {path} as {doc_id}")
    return IngestedDocument(
        document_id=doc_id,
        document_text=text,
        document_filename=os.path.basename(path),
        document_metadata={"file_size": os.path.getsize(path)},
    )


def _collect_documents(md_paths: list[str]) -> list[IngestedDocument]:
    docs: list[IngestedDocument] = []
    for path in md_paths:
        doc = _read_markdown(path)
        if doc:
            docs.append(doc)
    return docs


def _to_dataset(docs: list[IngestedDocument]) -> Dataset:
    dataset = Dataset.from_list([asdict(d) for d in docs])
    logger.debug(f"Constructed dataset with {len(dataset)} entries")
    return dataset


def run(config: dict[str, Any]) -> None:
    cfg = _load_config(config)
    if not cfg.run:
        logger.info("upload_ingest_to_hub stage is disabled")
        return

    source_dir = _resolve_source_dir(cfg, config)
    md_paths = sorted(glob.glob(os.path.join(source_dir, "*.md")))
    if not md_paths:
        raise FileNotFoundError(f"No .md files found in '{source_dir}'")

    documents = _collect_documents(md_paths)
    if not documents:
        raise FileNotFoundError(f"No valid markdown documents parsed in '{source_dir}'")

    dataset = _to_dataset(documents)
    custom_save_dataset(dataset=dataset, config=config, subset="ingested")
    logger.success("upload_ingest_to_hub stage complete")
