# RAG components

This package contains PDF ingestion, section retrieval, prompt construction,
and the retrieval evaluation script. The FastAPI endpoint and user interfaces
are documented in the [project README](../README.md).

The extracted section records retain titles and page numbers. They are marked
`unreviewed` and `approved_for_adopter_answers: false`; review and approve the
source guidance before using it for public adopter support.

`evaluation_questions.json` is a retrieval evaluation seed. It checks whether
expected sections appear in search results; it does not score generated answer
accuracy, language, or safety.
