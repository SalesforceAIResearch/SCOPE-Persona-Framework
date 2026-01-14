#!/usr/bin/env python3
"""
SCOPE: Shared Utilities

Common utilities for persona processing and evaluation.

Author: Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu
Paper: https://arxiv.org/abs/2601.07110
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load data from a JSON Lines file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """
    Save data to a JSON Lines file.

    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def parse_questionnaire_section(content: str, section_name: str) -> str:
    """
    Extract a specific section from questionnaire content.

    Args:
        content: Full questionnaire markdown content
        section_name: Name of section to extract

    Returns:
        Section content string
    """
    # Common section names
    all_sections = [
        "DEMOGRAPHIC INFORMATION",
        "SOCIODEMOGRAPHIC BEHAVIOR",
        "PERSONAL VALUES & MOTIVATIONS",
        "PERSONALITY TRAITS (Big Five)",
        "BEHAVIORAL PATTERNS & PREFERENCES",
        "PERSONAL IDENTITY & LIFE NARRATIVES",
        "PROFESSIONAL IDENTITY & CAREER",
        "CREATIVITY & INNOVATION"
    ]

    # Find this section's position
    try:
        section_idx = all_sections.index(section_name)
    except ValueError:
        return ""

    # Build boundary pattern
    if section_idx < len(all_sections) - 1:
        next_sections = all_sections[section_idx + 1:]
        boundary = rf"(?=## .*?(?:{'|'.join(re.escape(s) for s in next_sections)}))"
    else:
        boundary = r"(?=## .*QUESTIONNAIRE SUMMARY|\Z)"

    pattern = rf"## .*?{re.escape(section_name)}.*?\n(.*?){boundary}"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    return match.group(1).strip() if match else ""


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = answer.lower().strip()

    # Remove common prefixes
    prefixes = ['option', 'choice', 'answer:']
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()

    # Remove quotes
    normalized = normalized.strip('"\'')

    return normalized


def extract_numeric_scale(answer: str) -> Optional[int]:
    """
    Extract numeric value from scale answer.

    Args:
        answer: Answer string

    Returns:
        Integer value (1-5) or None
    """
    if not answer:
        return None

    # Direct numeric
    try:
        val = int(answer.strip())
        if 1 <= val <= 5:
            return val
    except ValueError:
        pass

    # Text mapping
    text_mapping = {
        'very inaccurate': 1, 'strongly disagree': 1, 'not at all': 1,
        'inaccurate': 2, 'disagree': 2, 'slightly': 2,
        'neutral': 3, 'neither': 3, 'moderately': 3,
        'accurate': 4, 'agree': 4, 'very': 4,
        'very accurate': 5, 'strongly agree': 5, 'extremely': 5
    }

    normalized = answer.lower().strip()
    for text, value in text_mapping.items():
        if text in normalized:
            return value

    return None


def calculate_similarity(response1: str, response2: str) -> float:
    """
    Calculate simple string similarity between two responses.

    Args:
        response1: First response
        response2: Second response

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not response1 or not response2:
        return 0.0

    # Normalize
    norm1 = normalize_answer(response1)
    norm2 = normalize_answer(response2)

    if norm1 == norm2:
        return 1.0

    # Simple word overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def format_persona_summary(persona: Dict, max_length: int = 500) -> str:
    """
    Create a brief summary of a persona.

    Args:
        persona: Persona dictionary
        max_length: Maximum character length

    Returns:
        Summary string
    """
    parts = []

    # Key demographic fields
    demo_fields = ['name', 'age', 'gender', 'occupation', 'location', 'education']
    for field in demo_fields:
        if field in persona and persona[field]:
            parts.append(f"{field.title()}: {persona[field]}")

    # Background/description
    for field in ['background', 'description', 'bio', 'persona_main']:
        if field in persona and persona[field]:
            desc = str(persona[field])[:200]
            parts.append(f"Background: {desc}...")
            break

    summary = " | ".join(parts)
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary


def get_facet_question_range(facet: str) -> tuple:
    """
    Get the question number range for a facet.

    Args:
        facet: Facet key name

    Returns:
        Tuple of (start_q, end_q) question numbers
    """
    ranges = {
        'demographics': (1, 13),
        'sociodemographic': (14, 50),
        'values': (51, 72),
        'personality': (73, 102),
        'behavioral': (103, 115),
        'identity': (116, 125),
        'professional': (126, 135),
        'creativity': (136, 141)
    }
    return ranges.get(facet.lower(), (0, 0))


def validate_persona_input(persona: Dict) -> List[str]:
    """
    Validate persona input and return any warnings.

    Args:
        persona: Persona dictionary

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    if not persona:
        warnings.append("Persona is empty")
        return warnings

    if not any(v for v in persona.values() if v):
        warnings.append("All persona fields are empty")

    # Check for minimal content
    total_content = sum(len(str(v)) for v in persona.values() if v)
    if total_content < 50:
        warnings.append("Persona has very little content (< 50 characters)")

    return warnings


class ResponseParser:
    """Parser for LLM responses to questionnaire format."""

    @staticmethod
    def parse_qa_format(response: str) -> Dict[str, str]:
        """
        Parse Q1: answer format responses.

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        responses = {}
        pattern = r'Q(\d+):\s*(.+?)(?=Q\d+:|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for q_num, answer in matches:
            responses[f"Q{q_num}"] = answer.strip()

        return responses

    @staticmethod
    def parse_numbered_format(response: str) -> Dict[str, str]:
        """
        Parse numbered list format (1. answer, 2. answer).

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        responses = {}
        pattern = r'(\d+)\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for q_num, answer in matches:
            responses[f"Q{q_num}"] = answer.strip()

        return responses

    @staticmethod
    def auto_parse(response: str) -> Dict[str, str]:
        """
        Automatically detect and parse response format.

        Args:
            response: LLM response text

        Returns:
            Dictionary mapping question numbers to answers
        """
        # Try Q format first
        if re.search(r'Q\d+:', response):
            return ResponseParser.parse_qa_format(response)

        # Try numbered format
        if re.search(r'^\d+\.', response, re.MULTILINE):
            return ResponseParser.parse_numbered_format(response)

        # Return empty if no pattern matched
        return {}
