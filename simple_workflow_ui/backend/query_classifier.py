#!/usr/bin/env python3
"""
Professional Query Classification using Cluster Embeddings Server
Leverages existing rp-node embeddings infrastructure - no local models needed
"""

import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple
import logging


class SemanticQueryClassifier:
    """Professional query classification using cluster embeddings server"""

    def __init__(self, embeddings_url: str = "http://192.168.1.81:9002"):
        # Use our existing embeddings server (load-balanced via HAProxy)
        self.embeddings_url = embeddings_url

        # Define example queries for each category (instead of keywords)
        self.category_examples = {
            "realtime_research": [
                "What is the current temperature in San Francisco?",
                "Latest news about AI developments",
                "Current stock price of Tesla",
                "What's happening right now in Ukraine?",
                "Today's weather forecast for New York",
                "Live sports scores NBA",
                "Breaking news about climate change",
                "Real-time traffic conditions in LA",
            ],
            "factual_research": [
                "What is the capital of France?",
                "Who invented the telephone?",
                "When was World War 2?",
                "Define machine learning",
                "History of the Roman Empire",
                "What is photosynthesis?",
                "Who was Albert Einstein?",
                "Explain quantum mechanics basics",
            ],
            "complex_research": [
                "Analyze the impact of AI on job markets",
                "Compare different renewable energy technologies",
                "Evaluate pros and cons of remote work",
                "Comprehensive analysis of blockchain technology",
                "Detailed comparison of programming languages",
                "Research the effects of social media on mental health",
                "Analyze economic trends in developing countries",
                "Compare different investment strategies",
            ],
            "general_research": [
                "Best restaurants in downtown Seattle",
                "How to learn Python programming",
                "Travel guide for Japan",
                "Healthy breakfast recipes",
                "Best laptops under $1000",
                "How to start a small business",
                "Benefits of meditation",
                "Popular dog breeds for families",
            ],
        }

        # Pre-compute embeddings for all examples using cluster server
        self.category_embeddings = self._compute_category_embeddings()
        logging.info(
            f"Initialized SemanticQueryClassifier with {len(self.category_examples)} categories"
        )

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from cluster embeddings server"""
        try:
            response = requests.post(
                f"{self.embeddings_url}/embeddings",
                json={"texts": texts, "model": "default"},
                timeout=30,
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"Embeddings server error: {e}")
            raise

    def _compute_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all category examples using cluster server"""
        category_embeddings = {}

        for category, examples in self.category_examples.items():
            try:
                # Get embeddings for all examples in this category from cluster server
                embeddings = self._get_embeddings(examples)
                # Use the mean embedding as the category centroid
                category_embeddings[category] = np.mean(embeddings, axis=0)
                logging.info(
                    f"Computed embeddings for '{category}' using {len(examples)} examples"
                )
            except Exception as e:
                logging.error(
                    f"Failed to compute embeddings for category '{category}': {e}"
                )
                # Fallback: use random embeddings (384 dimensions for all-MiniLM-L6-v2)
                category_embeddings[category] = np.random.normal(0, 0.1, 384)

        return category_embeddings

    def classify_query(self, query: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Classify query using semantic similarity

        Args:
            query: User query to classify
            threshold: Minimum confidence threshold

        Returns:
            Tuple of (category, confidence_score)
        """
        try:
            # Get embedding for the input query from cluster server
            query_embedding = self._get_embeddings([query])[0]

            # Calculate similarity with each category
            similarities = {}
            for category, category_embedding in self.category_embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), category_embedding.reshape(1, -1)
                )[0][0]
                similarities[category] = similarity

            # Get the best match
            best_category = max(similarities, key=similarities.get)
            best_score = similarities[best_category]

            # If confidence is too low, default to general research
            if best_score < threshold:
                return "general_research", best_score

            logging.info(
                f"Query classified as '{best_category}' with confidence {best_score:.3f}"
            )
            return best_category, best_score

        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return "general_research", 0.0

    def get_classification_details(self, query: str) -> Dict[str, float]:
        """Get detailed similarity scores for all categories"""
        query_embedding = self._get_embeddings([query])[0]

        similarities = {}
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), category_embedding.reshape(1, -1)
            )[0][0]
            similarities[category] = round(similarity, 3)

        return similarities

    def add_training_example(self, category: str, example: str):
        """
        Add a new training example and recompute embeddings
        This allows the classifier to learn from user feedback
        """
        if category in self.category_examples:
            self.category_examples[category].append(example)
            # Recompute embeddings for this category using cluster server
            examples = self.category_examples[category]
            try:
                embeddings = self._get_embeddings(examples)
                self.category_embeddings[category] = np.mean(embeddings, axis=0)
                logging.info(f"Added training example to '{category}': {example}")
            except Exception as e:
                logging.error(f"Failed to recompute embeddings for '{category}': {e}")
        else:
            logging.warning(f"Unknown category '{category}' - example not added")


# Global classifier instance (singleton pattern)
_classifier_instance = None


def get_query_classifier() -> SemanticQueryClassifier:
    """Get the global query classifier instance (singleton)"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SemanticQueryClassifier()
    return _classifier_instance
