from typing import List, Tuple, Optional, Dict
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

class Memory:
    def __init__(self):
        self.lines: List[str] = [""] * 1000000  # Initialize memory with 1 million lines
        # Initialize system memory layout
        self.lines[0] = "0"  # Instruction pointer
        self.lines[1] = ""   # Return value
        self.lines[2] = "0"  # Context pointer
        self.lines[3] = "[]" # Interrupt table

        # Load models for semantic matching
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.embeddings_cache: Dict[int, np.ndarray] = {}  # Cache embeddings for performance

    def read(self, address: str) -> str:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                if col < len(line_content):
                    return line_content[col]
                else:
                    return ""
            else:
                return ""
        else:
            if line < len(self.lines):
                return self.lines[line]
            else:
                return ""

    def write(self, address: str, text: str) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line >= len(self.lines):
                self.lines.extend([""] * (line - len(self.lines) + 1))
            line_content = self.lines[line]
            if len(line_content) <= col:
                line_content = line_content.ljust(col)
            line_content = line_content[:col] + text + line_content[col + len(text):]
            self.lines[line] = line_content
        else:
            if line >= len(self.lines):
                self.lines.extend([""] * (line - len(self.lines) + 1))
            self.lines[line] = text

    def insert(self, address: str, text: str) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                line_content = line_content[:col] + text + line_content[col:]
                self.lines[line] = line_content
            else:
                self.lines.extend([""] * (line - len(self.lines)))
                self.lines[line] = text
        else:
            if line <= len(self.lines):
                self.lines.insert(line, text)
            else:
                self.lines.extend([""] * (line - len(self.lines)))
                self.lines.append(text)

    def delete(self, address: str, length: int = 1) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                line_content = line_content[:col] + line_content[col + length:]
                self.lines[line] = line_content
        else:
            if line < len(self.lines):
                del self.lines[line]

    def parse_address(self, address: str) -> Tuple[int, Optional[int]]:
        if ":" in address:
            line_str, col_str = address.split(":")
            line = int(line_str)
            col = int(col_str)
            return line, col
        else:
            line = int(address)
            return line, None

    def append(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines) - 1  # Return the address of the appended line

    def resolve_pointer(self, pointer_str: str) -> str:
        # Implement pointer functions
        if pointer_str.startswith("exact_match("):
            match = re.match(r'exact_match\("(.+?)"(?:,\s*occurrence=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                occurrence = int(match.group(2)) if match.group(2) else 0
                return self.exact_match(query, occurrence)
        elif pointer_str.startswith("regex_match("):
            match = re.match(r'regex_match\("(.+?)"(?:,\s*occurrence=(\d+))?\)', pointer_str)
            if match:
                regex = match.group(1)
                occurrence = int(match.group(2)) if match.group(2) else 0
                return self.regex_match(regex, occurrence)
        elif pointer_str.startswith("tfidf_match("):
            match = re.match(r'tfidf_match\("(.+?)"(?:,\s*k=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                k = int(match.group(2)) if match.group(2) else 1
                return self.tfidf_match(query, k)
        elif pointer_str.startswith("semantic_match("):
            match = re.match(r'semantic_match\("(.+?)"(?:,\s*k=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                k = int(match.group(2)) if match.group(2) else 1
                return self.semantic_match(query, k)
        elif pointer_str.startswith("meta_pointer("):
            match = re.match(r'meta_pointer\((.+?)\)', pointer_str)
            if match:
                inner_pointer = match.group(1)
                resolved_pointer = self.resolve_pointer(inner_pointer)
                return resolved_pointer
        else:
            # Invalid pointer
            return "0"

    def exact_match(self, query: str, occurrence: int = 0) -> str:
        matches = [i for i, line in enumerate(self.lines) if line == query]
        if len(matches) > occurrence:
            return str(matches[occurrence])
        else:
            return "0"

    def regex_match(self, regex: str, occurrence: int = 0) -> str:
        pattern = re.compile(regex)
        matches = [i for i, line in enumerate(self.lines) if pattern.search(line)]
        if len(matches) > occurrence:
            return str(matches[occurrence])
        else:
            return "0"

    def tfidf_match(self, query: str, k: int = 1) -> str:
        vectorizer = TfidfVectorizer()
        documents = self.lines
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vec = vectorizer.transform([query])
        cosine_similarities = np.dot(tfidf_matrix, query_vec.T).toarray().ravel()
        top_indices = cosine_similarities.argsort()[-k:][::-1]
        if len(top_indices) > 0:
            return str(top_indices[0])
        else:
            return "0"

    def semantic_match(self, query: str, k: int = 1) -> str:
        query_embedding = self.get_embedding(query)
        similarities = []
        for idx, line in enumerate(self.lines):
            line_embedding = self.get_embedding(line)
            if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(line_embedding) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_embedding, line_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(line_embedding))
            similarities.append((idx, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        if len(similarities) > 0:
            return str(similarities[0][0])
        else:
            return "0"

    def get_embedding(self, text: str) -> np.ndarray:
        # Check cache
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        self.embeddings_cache[text_hash] = embedding
        return embedding
