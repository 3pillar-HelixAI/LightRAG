"""Deleting a document keeps the chunks another document shares.

A chunk id is a hash of the chunk's text alone, so an unchanged section of an
edited file, or the same file uploaded twice under different names, is one
chunk listed by both documents. Deleting the old version used to delete that
chunk too, and the surviving document stayed "processed" while part or all of
it could no longer be retrieved.
"""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline

SHARED = "Section shared by both versions."


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _embed(texts: list[str], **kwargs) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _llm(*args, **kwargs) -> str:
    return "ok"


def _chunk_by_section(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    """One chunk per blank-line separated section, so two documents can share one."""
    sections = [s.strip() for s in content.split("\n\n") if s.strip()]
    return [
        {"tokens": len(s), "content": s, "chunk_order_index": i}
        for i, s in enumerate(sections)
    ]


async def _build_rag(tmp_path, name: str) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / name),
        workspace=f"{name}_{uuid4().hex[:8]}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=_embed),
        tokenizer=Tokenizer("char-tokenizer", _CharTokenizer()),
        chunking_func=_chunk_by_section,
    )
    await rag.initialize_storages()
    return rag


async def _insert(rag: LightRAG, content: str, file_path: str) -> str:
    await rag.ainsert(content, file_paths=[file_path])
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    (doc_id,) = [i for i, s in docs.items() if s.file_path == file_path]
    return doc_id


async def _chunks_of(rag: LightRAG, doc_id: str) -> list[str]:
    status = await rag.doc_status.get_by_id(doc_id)
    return list(status["chunks_list"])


async def _stored(rag: LightRAG, chunk_ids: list[str]) -> dict[str, tuple[bool, bool]]:
    """chunk id -> (in text_chunks, in chunks_vdb)."""
    texts = await rag.text_chunks.get_by_ids(chunk_ids)
    vectors = await rag.chunks_vdb.get_by_ids(chunk_ids)
    return {
        chunk_id: (texts[i] is not None, vectors[i] is not None)
        for i, chunk_id in enumerate(chunk_ids)
    }


@pytest.mark.asyncio
async def test_delete_keeps_a_chunk_another_document_shares(tmp_path):
    rag = await _build_rag(tmp_path, "shared_section")
    try:
        old = await _insert(
            rag, f"{SHARED}\n\nOld wording of section two.", "spec_v1.md"
        )
        new = await _insert(
            rag, f"{SHARED}\n\nNew wording of section two.", "spec_v2.md"
        )
        shared_id = compute_mdhash_id(SHARED, prefix="chunk-")
        old_only = compute_mdhash_id("Old wording of section two.", prefix="chunk-")
        assert shared_id in await _chunks_of(rag, old)
        assert shared_id in await _chunks_of(rag, new)

        result = await rag.adelete_by_doc_id(old)

        assert result.status == "success"
        assert await rag.doc_status.get_by_id(old) is None
        # The surviving document can still retrieve every one of its chunks.
        new_chunks = await _chunks_of(rag, new)
        assert all(
            text and vector
            for text, vector in (await _stored(rag, new_chunks)).values()
        )
        # The chunk only the deleted document had is gone, as before.
        assert (await _stored(rag, [old_only]))[old_only] == (False, False)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_a_document_whose_chunks_are_all_shared(tmp_path):
    rag = await _build_rag(tmp_path, "all_shared")
    try:
        content = f"{SHARED}\n\nAnother section both copies have."
        first = await _insert(rag, content, "copy_a.md")
        second = await _insert(rag, content, "copy_b.md")
        assert first != second
        assert sorted(await _chunks_of(rag, first)) == sorted(
            await _chunks_of(rag, second)
        )

        result = await rag.adelete_by_doc_id(first)

        assert result.status == "success"
        assert "shared" in result.message
        assert await rag.doc_status.get_by_id(first) is None
        assert await rag.full_docs.get_by_id(first) is None
        assert await rag.full_entities.get_by_id(first) is None
        assert await rag.full_relations.get_by_id(first) is None
        second_chunks = await _chunks_of(rag, second)
        assert all(
            text and vector
            for text, vector in (await _stored(rag, second_chunks)).values()
        )
        assert (await rag.doc_status.get_by_id(second))["status"] == DocStatus.PROCESSED
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_an_entity_sourced_from_a_shared_chunk_survives(tmp_path):
    rag = await _build_rag(tmp_path, "shared_entity")
    try:
        old = await _insert(rag, f"{SHARED}\n\nOld only.", "plan_v1.md")
        new = await _insert(rag, f"{SHARED}\n\nNew only.", "plan_v2.md")
        shared_id = compute_mdhash_id(SHARED, prefix="chunk-")
        # An entity extracted from the shared section, as both documents' graph index records it.
        entity = "SHARED-ENTITY"
        await rag.chunk_entity_relation_graph.upsert_node(
            entity,
            {
                "entity_id": entity,
                "source_id": shared_id,
                "description": "named in the shared section",
                "entity_type": "test",
                "file_path": "plan_v1.md",
                "created_at": int(datetime.now(timezone.utc).timestamp()),
            },
        )
        await rag.entity_chunks.upsert({entity: {"chunk_ids": [shared_id], "count": 1}})
        for doc_id in (old, new):
            await rag.full_entities.upsert({doc_id: {"entity_names": [entity]}})

        result = await rag.adelete_by_doc_id(old)

        assert result.status == "success"
        node = await rag.chunk_entity_relation_graph.get_node(entity)
        assert node is not None
        assert shared_id in node["source_id"].split(GRAPH_FIELD_SEP)
        tracking = await rag.entity_chunks.get_by_id(entity)
        assert tracking is not None and shared_id in tracking["chunk_ids"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_document_with_no_shared_chunks_is_deleted_entirely(tmp_path):
    rag = await _build_rag(tmp_path, "unshared")
    try:
        doc = await _insert(rag, "Only here.\n\nAlso only here.", "alone.md")
        other = await _insert(rag, "Unrelated text.", "other.md")
        doc_chunks = await _chunks_of(rag, doc)

        result = await rag.adelete_by_doc_id(doc)

        assert result.status == "success"
        assert all(
            stored == (False, False)
            for stored in (await _stored(rag, doc_chunks)).values()
        )
        other_chunks = await _chunks_of(rag, other)
        assert all(
            text and vector
            for text, vector in (await _stored(rag, other_chunks)).values()
        )
    finally:
        await rag.finalize_storages()
