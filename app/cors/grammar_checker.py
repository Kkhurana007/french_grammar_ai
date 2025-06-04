"""
Main Grammar Checker Service
Combines NLP processing, rule validation, and AI suggestions
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy
from transformers import pipeline
from core.rules import FrenchGrammarRules, GrammarError, RuleType
from app.common.utils import analyze_sentence_structure
from config import settings

@dataclass
class GrammarCheckResult:
    """Structured result of grammar checking"""
    original_text: str
    errors: List[GrammarError]
    structure: List[Dict]
    ai_suggestions: List[str]
    stats: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Convert result to API-friendly dictionary"""
        return {
            "text": self.original_text,
            "error_count": len(self.errors),
            "errors": [{
                "type": error.rule_type.french_label(),
                "message": error.message,
                "suggestion": error.suggestion,
                "context": error.highlighted,
                "start_pos": error.start_pos,
                "end_pos": error.end_pos
            } for error in self.errors],
            "structure": self.structure,
            "ai_suggestions": self.ai_suggestions,
            "stats": self.stats
        }

class FrenchGrammarChecker:
    """
    Main grammar checking service with multi-layer validation:
    1. Rule-based grammar checking
    2. AI-powered suggestions
    3. Structure analysis
    """
    
    def __init__(self):
        # Initialize rule-based checker
        self.rule_checker = FrenchGrammarRules(settings.SPACY_MODEL)
        
        # Load AI models
        self.ai_grammar_model = pipeline(
            "text2text-generation",
            model=settings.GRAMMAR_MODEL_NAME,
            cache_dir=settings.TRANSFORMERS_CACHE
        )
        
        # Performance optimization
        self._nlp = spacy.load(settings.SPACY_MODEL)
    
    def check(self, text: str) -> GrammarCheckResult:
        """
        Perform comprehensive grammar checking
        Args:
            text: Input French text to check
        Returns:
            GrammarCheckResult with all findings
        """
        # Basic validation
        if not text.strip():
            return self._empty_result(text)
        
        # Rule-based checking
        errors = self.rule_checker.check(text)
        
        # AI-powered suggestions
        ai_suggestions = self._get_ai_suggestions(text, errors)
        
        # Structure analysis
        structure = analyze_sentence_structure(text)
        
        # Generate statistics
        stats = self._generate_stats(errors)
        
        return GrammarCheckResult(
            original_text=text,
            errors=errors,
            structure=structure,
            ai_suggestions=ai_suggestions,
            stats=stats
        )
    
    def _get_ai_suggestions(self, text: str, errors: List[GrammarError]) -> List[str]:
        """
        Get AI-powered suggestions for the text
        Args:
            text: Original text
            errors: Detected rule-based errors
        Returns:
            List of improvement suggestions
        """
        try:
            # Only query AI if there are errors
            if not errors:
                return []
            
            # Create focused prompts based on errors
            prompts = [
                f"Corrigez cette phrase en français en expliquant la règle: {text}",
                f"Suggestions d'amélioration pour: {text}"
            ]
            
            results = []
            for prompt in prompts:
                output = self.ai_grammar_model(
                    prompt,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7
                )
                results.append(output[0]["generated_text"])
            
            return results
        
        except Exception as e:
            print(f"AI suggestion failed: {str(e)}")
            return []
    
    def _generate_stats(self, errors: List[GrammarError]) -> Dict[str, int]:
        """Generate statistics about errors"""
        stats = {
            "total_errors": len(errors),
            "by_type": {rule_type.french_label(): 0 for rule_type in RuleType}
        }
        
        for error in errors:
            stats["by_type"][error.rule_type.french_label()] += 1
        
        return stats
    
    def _empty_result(self, text: str) -> GrammarCheckResult:
        """Return empty result for empty input"""
        return GrammarCheckResult(
            original_text=text,
            errors=[],
            structure=[],
            ai_suggestions=[],
            stats={"total_errors": 0, "by_type": {}}
        )
    
    def batch_check(self, texts: List[str]) -> List[GrammarCheckResult]:
        """Process multiple texts efficiently"""
        docs = list(self._nlp.pipe(texts))
        return [self.check(doc.text) for doc in docs]

# Example usage
if __name__ == "__main__":
    checker = FrenchGrammarChecker()
    sample = "Je souvent mange du fromage."
    
    result = checker.check(sample)
    print(f"Found {len(result.errors)} errors:")
    for error in result.errors:
        print(f"- {error.rule_type.french_label()}: {error.message}")
        print(f"  Suggested: {error.suggestion}")
    
    print("\nAI Suggestions:")
    for suggestion in result.ai_suggestions:
        print(f"- {suggestion}")