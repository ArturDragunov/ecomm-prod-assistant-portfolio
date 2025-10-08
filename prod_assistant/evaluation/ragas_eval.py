import asyncio
from utils.model_loader import ModelLoader
from ragas import SingleTurnSample # ragas is similar to deepeval -> for evaluation of RAG systems
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
import grpc.experimental.aio as grpc_aio # for async calling
grpc_aio.init_grpc_aio()
model_loader=ModelLoader()

# 📊 High-level comparison
# Metric	What it Evaluates	What it Tells You	Uses LLM?	Uses Embeddings?
# Context Precision	Quality of retrieved documents	“Did we fetch the right information?”	✅	❌
# Response Relevancy	Quality of generated answer	“Did the model use the right info?”	    ✅	✅

# we add async modules to the code, because ragas perforns async calls to the LLM and embeddings.
# You need asyncio here because:
# ragas (and possibly grpc.experimental.aio) exposes async methods that must be awaited.
# You’re mixing sync outer functions with async inner calls.
# asyncio.run() bridges that gap.
def evaluate_context_precision(query, response, retrieved_context):
    """“Did the retriever bring in useful and focused information for the query and response?”
        It measures how much of the retrieved context was actually useful for answering the question, according to the LLM evaluator.
        So even if you retrieved 5 paragraphs, if only 1 is relevant, your context precision will be low.
    """
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        # we don't know exact answer (no reference). We evaluate the query, llm response, and retrieved context
        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await context_precision.single_turn_ascore(sample) # requires async calling
            return result
        # since your top-level evaluate_context_precision() is a normal function (not async), you can’t await directly in it.
        # that's why you need to call asyncio.run(main())
        return asyncio.run(main())
    except Exception as e:
        return e

def evaluate_response_relevancy(query, response, retrieved_context):
    """“Is the model’s generated response relevant and supported by the retrieved context?”
        This measures how well the LLM’s output aligns with the retrieved evidence.
        Even if the context precision was good (retriever found great docs), the generator might:
            • hallucinate facts,
            • drift from the context, or
            • ignore the retrieved info altogether.
        This metric penalizes that.
    """
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            embedding_model = model_loader.load_embeddings()
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
            scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
            result = await scorer.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        return e