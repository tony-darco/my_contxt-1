from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi



class myembed:
    embedding_model: OllamaEmbeddings
    vector_store: Chroma
    all_documents: list[Document]
    bm25: BM25Okapi

    def __init__(self) -> None:
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large", base_url='http://192.168.1.17:11434/')

        self.vector_store = Chroma(
            collection_name="cisco360",
            embedding_function=self.embedding_model,
            persist_directory='chromadbcontxt1'
        )

        self.all_documents: list[Document] = []

        if self.vector_store._collection.count() == 0:
            self.store()
        else:
            results = self.vector_store.get()
            self.all_documents = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(results['documents'], results['metadatas'])
            ]

    def store(self) -> None:

        pdf_files: list[dict[str, str]] = [
            {
                "path": "360-partner-program-partner-value-index-cisco-partner-incentive-metrics-guide.pdf",
                "title": "Cisco 360 Partner Program - Partner Value Index & Incentive Metrics Guide",
                "category": "incentive-metrics",
            },
            {
                "path": "cisco-360-program-partner-faq.pdf",
                "title": "Cisco 360 Program Partner FAQ",
                "category": "faq",
            },
        ]
    
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

        for pdf in pdf_files:
            loader: PyPDFLoader = PyPDFLoader(pdf["path"])
            pages: list[Document] = loader.load()
            chunks: list[Document] = text_splitter.split_documents(pages)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": pdf["path"],
                    "title": pdf["title"],
                    "category": pdf["category"],
                    "program": "cisco-360",
                    "chunk_index": i,
                })
            self.all_documents.extend(chunks)

        self.vector_store.add_documents(self.all_documents)

    def get_store(self) -> Chroma:
        return self.vector_store

    def rank25(self) -> BM25Okapi:
        tokenized_corpus: list[list[str]] = [doc.page_content.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self.bm25

    async def search_bm25(self, query: str, k: int = 4) -> list[Document]:
        query_tokens: list[str] = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-k:][::-1]
        return [self.all_documents[i] for i in top_indices]

    async def search_vstore(self, query: str, k: int = 4) -> list[Document]:
        return await self.vector_store.asimilarity_search(query=query, k=k)

