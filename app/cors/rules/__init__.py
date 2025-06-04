"""
French Grammar Rules Module
Central registry for all grammar rules and validation logic
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import spacy
from spacy.tokens import Token, Doc

# Import all rule modules
from .adverb_rules import AdverbRules
from .conjugation_rules import ConjugationRules
from .agreement_rules import AgreementRules
from .sentence_structure_rules import SentenceStructureRules

class RuleType(Enum):
    """Categories of grammar rules"""
    ADVERB = "adverb_placement"
    CONJUGATION = "verb_conjugation"
    AGREEMENT = "grammatical_agreement"
    STRUCTURE = "sentence_structure"
    PUNCTUATION = "punctuation"

@dataclass
class GrammarError:
    """Structure for grammar error reporting"""
    rule_type: RuleType
    message: str
    suggestion: str
    context: str
    start_pos: int
    end_pos: int

class FrenchGrammarRules:
    """
    Main class that orchestrates all grammar rules
    Usage:
        rules = FrenchGrammarRules()
        errors = rules.check_sentence("Je vais au parc")
    """
    
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_lg")
        self.rules = {
            RuleType.ADVERB: AdverbRules(),
            RuleType.CONJUGATION: ConjugationRules(),
            RuleType.AGREEMENT: AgreementRules(),
            RuleType.STRUCTURE: SentenceStructureRules()
        }
    
    def check_sentence(self, sentence: str) -> List[GrammarError]:
        """
        Validate a French sentence against all grammar rules
        Args:
            sentence: Input sentence to check
        Returns:
            List of GrammarError objects found
        """
        doc = self.nlp(sentence)
        errors = []
        
        for rule_type, rule_instance in self.rules.items():
            errors.extend(rule_instance.apply(doc))
        
        return errors
    
    def get_rule(self, rule_type: RuleType):
        """Access specific rule category"""
        return self.rules.get(rule_type)

# Export the main interface
__all__ = ['FrenchGrammarRules', 'GrammarError', 'RuleType']