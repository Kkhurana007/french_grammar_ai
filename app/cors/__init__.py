"""
French Grammar Rules Module
Central registry and interface for all grammar validation rules
"""

from __future__ import annotations
from typing import Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum, auto
import spacy
from spacy.tokens import Token, Doc

class RuleType(Enum):
    """Categories of grammar rules with French labels"""
    ADVERB = auto()
    CONJUGATION = auto()
    AGREEMENT = auto()
    STRUCTURE = auto()
    PUNCTUATION = auto()
    
    def french_label(self) -> str:
        labels = {
            RuleType.ADVERB: "Place des adverbes",
            RuleType.CONJUGATION: "Conjugaison",
            RuleType.AGREEMENT: "Accord",
            RuleType.STRUCTURE: "Structure de phrase",
            RuleType.PUNCTUATION: "Ponctuation"
        }
        return labels[self]

@runtime_checkable
class GrammarRule(Protocol):
    """Protocol for all grammar rule implementations"""
    def apply(self, doc: Doc) -> List[GrammarError]: ...
    
    @property
    def rule_type(self) -> RuleType: ...

@dataclass
class GrammarError:
    """Detailed grammar error representation"""
    rule_type: RuleType
    message: str
    suggestion: str
    context: str
    start_pos: int
    end_pos: int
    source_text: str
    
    @property
    def highlighted(self) -> str:
        """Generate highlighted error context"""
        before = self.source_text[:self.start_pos]
        error = self.source_text[self.start_pos:self.end_pos]
        after = self.source_text[self.end_pos:]
        return f"{before}>>>{error}<<<{after}"

class FrenchGrammarRules:
    """
    Main grammar rule orchestrator with caching and configuration
    Example:
        >>> rules = FrenchGrammarRules()
        >>> errors = rules.check("Je souvent mange")
    """
    
    def __init__(self, nlp_model: Optional[str] = None):
        self.nlp = spacy.load(nlp_model or "fr_core_news_lg")
        self._rules: Dict[RuleType, GrammarRule] = self._load_rules()
    
    def _load_rules(self) -> Dict[RuleType, GrammarRule]:
        """Initialize all rule implementations"""
        # Import here to prevent circular imports
        from .adverb_rules import AdverbRules
        from .conjugation_rules import ConjugationRules
        from .agreement_rules import AgreementRules
        from .sentence_structure_rules import SentenceStructureRules
        
        return {
            RuleType.ADVERB: AdverbRules(),
            RuleType.CONJUGATION: ConjugationRules(),
            RuleType.AGREEMENT: AgreementRules(),
            RuleType.STRUCTURE: SentenceStructureRules()
        }
    
    def check(self, text: str) -> List[GrammarError]:
        """Validate French text against all registered rules"""
        doc = self.nlp(text)
        errors: List[GrammarError] = []
        
        for rule in self._rules.values():
            try:
                for error in rule.apply(doc):
                    error.source_text = text  # Attach original text
                    errors.append(error)
            except Exception as e:
                print(f"Error applying {rule.rule_type} rules: {str(e)}")
        
        return sorted(errors, key=lambda e: e.start_pos)
    
    def get_rule(self, rule_type: RuleType) -> Optional[GrammarRule]:
        """Access specific rule category"""
        return self._rules.get(rule_type)
    
    def add_rule(self, rule: GrammarRule) -> None:
        """Register a custom rule implementation"""
        if isinstance(rule, GrammarRule):
            self._rules[rule.rule_type] = rule

# Export public interface
__all__ = [
    'FrenchGrammarRules',
    'GrammarError',
    'RuleType',
    'GrammarRule'
]