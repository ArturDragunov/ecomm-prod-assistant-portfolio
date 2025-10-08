import os
from typing import List
from langchain_astradb import AstraDBVectorStore
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_core.documents import Document
# Add the project root to the Python path for direct script execution
# project_root = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(project_root))

class Retriever:
    def __init__(self):
        """_summary_
        """
        load_dotenv()
        self.model_loader=ModelLoader()
        self.config=load_config()
        self.vstore = None
        self.retriever_instance = None
    
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def load_retriever(self):
        """_summary_
        """
        if not self.vstore:
            # if vector store is not loaded, load it from AstraDB
            collection_name = self.config["astra_db"]["collection_name"]
            
            self.vstore =AstraDBVectorStore(
                embedding= self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
                )
        if not self.retriever_instance:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            # base_retriever (mmr_retriever) — fetch fetch_k candidates from the vector DB using vector similarity,
            #  then apply MMR to select a final set of k documents balancing relevance & diversity.
            mmr_retriever=self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k,             # final number of items returned. The number of documents you want back after MMR.
                                "fetch_k": 20,          # number of initial candidates fetched from vector DB BEFORE applying MMR.
                                "lambda_mult": 0.7,     # MMR trade-off parameter. λ ≈ 1.0 → favor relevance (behaves like plain top-k similarity).
                                                        # λ ≈ 0.0 → favor diversity (select very different documents even if less similar).
                                "score_threshold": 0.6  # filter threshold. only consider docs with similarity ≥ 0.6
                                })
            print("Retriever loaded successfully.")
            
            llm = self.model_loader.load_llm()
            
            # base_compressor (LLMChainFilter) — for each selected document, run an LLM-based compression/filter
            #  that extracts or compresses the parts of the document most relevant to the query.
            # for each document (or chunk of a document), it runs a prompt like “Given the user query and this document text, extract the sentences/paragraphs that are relevant to the query” — returning a shortened / focused page_content.
            # This can be extractive (return exact sentences) or abstractive (generate a concise summary).
            # It can also be used as a binary filter: keep vs discard.
            compressor=LLMChainFilter.from_llm(llm)
            
            # ContextualCompressionRetriever — orchestrates steps 1+2 and returns the compressed documents to the caller.
            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=mmr_retriever
            )
            
        return self.retriever_instance
            
    def call_retriever(self,query):
        """_summary_
        """
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output
    
if __name__=='__main__':
    user_query = "I want to buy a DELL laptop, would you recommend it?"
    
    retriever_obj = Retriever()
    
    retrieved_docs = retriever_obj.call_retriever(user_query)
    
    def _format_docs(docs: List[Document]) -> List[str]:
        if not docs:
            return ["No relevant documents found."]
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return formatted_chunks  # Return list instead of joined string
    
    retrieved_contexts = _format_docs(retrieved_docs)
    
    #this is not an actual output this have been written to test the pipeline
    response="DELL laptops have good reviews and price-value ratio"
    
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)
    
    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)
    

    
    
    
    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")