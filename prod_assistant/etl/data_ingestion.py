import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config

class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self): # methods with __ are called render methods
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config=load_config()

    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """
        load_dotenv() # making it compatible with both environments (local and k8s)
        
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        # if any env variable is missing (validated by required_vars), raise an error
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

       

    def _get_csv_path(self):
        """
        Get path to the CSV file located inside 'data' folder.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir,'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self):
        """
        Load product data from CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id','product_title', 'rating', 'total_reviews','price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product data into list of LangChain Document objects.
        """
        documents = []
        
        for _, row in self.product_data.iterrows():
            metadata = {
                "product_id": row["product_id"],
                "product_title": row["product_title"],
                "rating": row["rating"],
                "total_reviews": row["total_reviews"],
                "price": row["price"]
            } # review is content and the rest is metadata
            doc = Document(page_content=row["top_reviews"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

# We do our own Document loader from csv instead of using CSVLoader from langchain, because we want custom metadata
# CSVLoader puts everything to page_content, and metadata is just a path to the file
# ***Example***:
# Document(metadata={'source': 'D:\\\\complete_content_new\\\\llmops-batch\\\\ecomm-prod-assistant\\\\data\\\\product_reviews.csv', 'row': 1},
#  page_content='product_id: itm7579ed94ca647\n
# product_title: Apple iPhone 15 (Pink, 128 GB)\n
# rating: 4.6\ntotal_reviews: 9,407\n
# price: ₹64,900\n
# top_reviews: 5 Worth every penny Just go for it.
# Amazing one.Beautiful camera with super fast processor READ MORE bijaya mohanty Certified Buyer ,
#  Baleshwar May, 2024 4317 1058 Permalink Report Abuse || 5 Fabulous!
#  So beautiful, so elegant, just a vowww😍❤️ READ MORE Akshay Meena Certified Buyer , Jaipur Nov, 2023 897 202 Permalink Report Abuse')


    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into AstraDB vector store.
        """
        # collection name is the db name you defined for AstraDB
        collection_name=self.config["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding= self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        # return loaded vector store and inserted ids
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform_data()
        vstore, _ = self.store_in_vector_db(documents)

        #Optionally do a quick search
        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()