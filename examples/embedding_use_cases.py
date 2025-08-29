#!/usr/bin/env python3
"""
Examples of how embeddings enhance LangGraph workflows
Demonstrates why we need the embedding server
"""

import asyncio
import requests
import numpy as np
from typing import List, Dict, Any


class EmbeddingExamples:
    def __init__(self):
        self.embedding_url = "http://192.168.1.178:8081"  # rp-node
        self.vector_db = {}  # Simple in-memory store for examples

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from our local server"""
        response = requests.post(
            f"{self.embedding_url}/embeddings",
            json={"texts": texts, "model": "default"},
        )
        return response.json()["embeddings"]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate similarity between vectors"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def example_1_document_rag(self):
        """Example 1: RAG for local documentation"""
        print("🔍 Example 1: Document RAG (Retrieval Augmented Generation)")
        print("=" * 60)

        # Your local knowledge base
        local_docs = [
            "Jetson Orin Nano setup requires nvpmodel -m 0 for maximum performance",
            "HAProxy load balancer runs on cpu-node port 9000 for LLM requests",
            "Redis cache is configured on cpu-node with password langgraph_redis_pass",
            "Embedding server runs on rp-node using ARM-optimized sentence-transformers",
            "Tools server on worker-node3 provides web scraping and command execution",
            "Monitoring server on worker-node4 tracks cluster health every 30 seconds",
        ]

        # Create embeddings for all documents
        print("Creating embeddings for knowledge base...")
        doc_embeddings = self.embed_texts(local_docs)

        # Store in vector database
        for i, (doc, embedding) in enumerate(zip(local_docs, doc_embeddings)):
            self.vector_db[f"doc_{i}"] = {"text": doc, "embedding": embedding}

        # User question
        user_question = "How do I optimize Jetson performance?"
        question_embedding = self.embed_texts([user_question])[0]

        # Find most similar documents
        similarities = []
        for doc_id, doc_data in self.vector_db.items():
            similarity = self.cosine_similarity(
                question_embedding, doc_data["embedding"]
            )
            similarities.append((similarity, doc_data["text"]))

        # Sort by similarity
        similarities.sort(reverse=True)

        print(f"\nUser Question: '{user_question}'")
        print("\nMost relevant documents:")
        for i, (score, text) in enumerate(similarities[:3]):
            print(f"{i+1}. [Score: {score:.3f}] {text}")

        print(
            "\n💡 LangGraph workflow would then send the top documents + question to LLM"
        )

    def example_2_semantic_routing(self):
        """Example 2: Semantic routing for LangGraph"""
        print("\n🔀 Example 2: Semantic Routing in LangGraph")
        print("=" * 60)

        # Define different workflow types with their descriptions
        workflow_types = {
            "coding_help": "Programming questions, code generation, debugging, software development",
            "system_admin": "Server configuration, deployment, system monitoring, infrastructure",
            "research": "Information gathering, web search, data analysis, summarization",
            "general_chat": "Casual conversation, greetings, simple questions, small talk",
        }

        # Create embeddings for workflow descriptions
        descriptions = list(workflow_types.values())
        workflow_embeddings = self.embed_texts(descriptions)

        # Test user inputs
        test_inputs = [
            "How do I fix this Python error?",
            "What's the status of our HAProxy server?",
            "Search for the latest AI research papers",
            "Hello, how are you doing today?",
        ]

        input_embeddings = self.embed_texts(test_inputs)

        print("Routing user inputs to appropriate workflows:")
        print()

        for user_input, input_embedding in zip(test_inputs, input_embeddings):
            best_match = None
            best_score = -1

            for i, (workflow, desc) in enumerate(workflow_types.items()):
                score = self.cosine_similarity(input_embedding, workflow_embeddings[i])
                if score > best_score:
                    best_score = score
                    best_match = workflow

            print(f"Input: '{user_input}'")
            print(f"→ Route to: {best_match} (confidence: {best_score:.3f})")
            print()

    def example_3_memory_retrieval(self):
        """Example 3: Conversation memory with embeddings"""
        print("🧠 Example 3: Conversation Memory Retrieval")
        print("=" * 60)

        # Simulate conversation history
        conversation_history = [
            "I'm working on setting up a LangGraph cluster on my local network",
            "My Jetson Orin Nano is running at 192.168.1.177 with Ollama",
            "The cpu-node has 32GB RAM and runs the load balancer",
            "I want to optimize the embeddings server performance",
            "Redis is caching results on port 6379",
        ]

        # Create embeddings for conversation history
        history_embeddings = self.embed_texts(conversation_history)

        # Current user question
        current_question = "What was the IP address of my Jetson again?"
        question_embedding = self.embed_texts([current_question])[0]

        # Find relevant conversation context
        similarities = []
        for i, (msg, embedding) in enumerate(
            zip(conversation_history, history_embeddings)
        ):
            similarity = self.cosine_similarity(question_embedding, embedding)
            similarities.append((similarity, i, msg))

        similarities.sort(reverse=True)

        print(f"Current question: '{current_question}'")
        print("\nRelevant conversation context:")
        for score, idx, msg in similarities[:2]:
            print(f"[Score: {score:.3f}] Message {idx}: {msg}")

        print("\n💡 LangGraph can use this context to provide accurate answers")

    def example_4_tool_selection(self):
        """Example 4: Smart tool selection using embeddings"""
        print("\n🛠️ Example 4: Smart Tool Selection")
        print("=" * 60)

        # Available tools with descriptions
        available_tools = {
            "web_search": "Search the internet for information, news, articles, current events",
            "web_scrape": "Extract content from specific websites and web pages",
            "execute_command": "Run shell commands, system operations, file operations",
            "llm_query": "Generate text, answer questions, creative writing, analysis",
            "embeddings": "Create vector representations, similarity search, semantic analysis",
        }

        # Create embeddings for tool descriptions
        tool_descriptions = list(available_tools.values())
        tool_embeddings = self.embed_texts(tool_descriptions)

        # Test user requests
        user_requests = [
            "Find the latest news about artificial intelligence",
            "Get the content from https://example.com",
            "Check how much disk space is available",
            "Write a poem about machine learning",
            "Find documents similar to this text",
        ]

        request_embeddings = self.embed_texts(user_requests)

        print("Selecting appropriate tools for user requests:")
        print()

        for request, request_embedding in zip(user_requests, request_embeddings):
            best_tool = None
            best_score = -1

            for i, (tool_name, desc) in enumerate(available_tools.items()):
                score = self.cosine_similarity(request_embedding, tool_embeddings[i])
                if score > best_score:
                    best_score = score
                    best_tool = tool_name

            print(f"Request: '{request}'")
            print(f"→ Best tool: {best_tool} (confidence: {best_score:.3f})")
            print()


def main():
    """Run all embedding examples"""
    print("🎯 LangGraph Embedding Server Use Cases")
    print("🖥️ Demonstrating why we need rp-node (192.168.1.178:8081)")
    print("=" * 80)

    examples = EmbeddingExamples()

    try:
        # Test if embedding server is available
        response = requests.get(f"{examples.embedding_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Embedding server not available. Start it first!")
            return
    except:
        print("❌ Cannot connect to embedding server. Is rp-node running?")
        print(
            "   Start with: ssh sanzad@192.168.1.178 'sudo systemctl start embeddings-server'"
        )
        return

    print("✅ Embedding server is available. Running examples...")
    print()

    # Run all examples
    examples.example_1_document_rag()
    examples.example_2_semantic_routing()
    examples.example_3_memory_retrieval()
    examples.example_4_tool_selection()

    print("\n🎉 Examples completed!")
    print("\n📝 Key Benefits of Local Embedding Server:")
    print("   • RAG: Retrieve relevant documents for context")
    print("   • Routing: Intelligently route queries to appropriate workflows")
    print("   • Memory: Recall relevant conversation history")
    print("   • Tools: Select best tools based on semantic understanding")
    print("   • Privacy: All semantic processing stays local")
    print("   • Zero Cost: No external API calls for embeddings")


if __name__ == "__main__":
    main()
