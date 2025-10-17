from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph + MCP (Retriever + WebSearch)."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        
    async def async_init(self):
        self.mcp_tools = await self.mcp_client.get_tools()

    def __init__(self):
        self.model_loader = ModelLoader()
        self.retriever_obj = Retriever(self.model_loader)
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # MCP Client Init
        self.mcp_client = MultiServerMCPClient({
            "hybrid_search": {
                "command": "python",
                "args": ["prod_assistant/mcp_servers/product_search_server.py"],
                "transport": "stdio",
            }
        })
        self.mcp_tools = None  # Will be loaded in async_init
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    async def _vector_retriever(self, state: AgentState):
        # you can't await inside a sync function, thus we use async def
        # The async approach (File 2) is superior because:
        # Non-blocking - doesn't freeze the workflow
        # Better resource usage - doesn't create new event loops
        # Consistent with LangGraph - LangGraph expects async nodes
        # Scalable - multiple async operations can run concurrently
        print("--- RETRIEVER (MCP) ---")
        query = state["messages"][-1].content
        tool = next(t for t in self.mcp_tools if t.name == "get_product_info")
        result= await tool.ainvoke({"query": query})
        context = result if result else "No data"
        return {"messages": [HumanMessage(content=context)]}

    def _web_search(self, state: AgentState):
        print("--- WEB SEARCH (MCP) ---")
        query = state["messages"][-1].content
        tool = next(t for t in self.mcp_tools if t.name == "web_search")
        result = asyncio.run(tool.ainvoke({"query": query}))
        context = result if result else "No data from web"
        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine. "
            "Do NOT answer the query. Only rewrite it.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        new_q = chain.invoke({"question": question})
        return {"messages": [HumanMessage(content=new_q.strip())]}


    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            
            # lambda anon function. Syntax: <parameter/input>: <expression/return value>
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END, # if assistant says TOOL, then I go to retriever tool
            
            { # from assistant either I go to retriever tool or I end the process
                "Retriever": "Retriever", 
                 END: END
             },
        )
        workflow.add_conditional_edges(
            
            "Retriever",
            
            self._grade_documents,
            
            {"generator": "Generator", # if the documents are relevant (returned yes), then I go to generator
             
             "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        
        workflow.add_edge("Rewriter", "WebSearch")
        
        workflow.add_edge("WebSearch", "Generator")
        
        return workflow

    # ---------- Public Run ----------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        # Initialize MCP tools if not already done
        if self.mcp_tools is None:
            await self.async_init()
        
        result = await self.app.ainvoke({"messages": [HumanMessage(content=query)]},
                                        config={"configurable": {"thread_id": thread_id}})
        return result["messages"][-1].content


if __name__ == "__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 16?")
    print("\nFinal Answer:\n", answer)

# You can use either lambda for conditional edges or actual functions.
# def route_to_retriever_or_end(state):
#     last_message_content = state["messages"][-1].content
#     if "TOOL" in last_message_content:
#         return "Retriever"
#     else:
#         return END

# # Usage in LangGraph
# .add_conditional_edges(
#     "some_node",
#     route_to_retriever_or_end,  # Instead of lambda
#     {"Retriever": "Retriever", END: END}
# )