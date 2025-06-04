"""
Tests for French Grammar Rules API
Validates both rule implementations and their integration
"""

import pytest
from spacy.tokens import Doc
from core.rules import FrenchGrammarRules, GrammarError, RuleType
from core.rules.adverb_rules import AdverbRules
from core.rules.agreement_rules import AgreementRules
from app.common.utils import analyze_sentence_structure

# Sample sentences for testing
TEST_CASES = [
    # (sentence, expected_error_types, should_pass)
    ("Je mange rapidement.", [], True),  # Correct
    ("Je rapidement mange.", [RuleType.ADVERB], False),  # Adverb error
    ("La fille belle.", [RuleType.AGREEMENT], False),  # Agreement error
    ("Il vont au parc.", [RuleType.CONJUGATION], False),  # Conjugation error
]

@pytest.fixture
def grammar_rules():
    return FrenchGrammarRules()

@pytest.fixture
def sample_doc(grammar_rules):
    return grammar_rules.nlp("Je souvent mange des pommes.")

class TestGrammarAPI:
    @pytest.mark.parametrize("sentence,expected_errors,should_pass", TEST_CASES)
    def test_sentence_validation(self, grammar_rules, sentence, expected_errors, should_pass):
        """Test end-to-end sentence validation"""
        errors = grammar_rules.check_sentence(sentence)
        
        if should_pass:
            assert len(errors) == 0, f"Expected no errors but got: {errors}"
        else:
            assert len(errors) > 0, "Expected errors but found none"
            assert all(e.rule_type in expected_errors for e in errors), (
                f"Unexpected error types. Got {[e.rule_type for e in errors]}, "
                f"expected {expected_errors}"
            )

    def test_error_structure(self, grammar_rules):
        """Validate GrammarError dataclass structure"""
        errors = grammar_rules.check_sentence("Je souvent mange.")
        assert errors, "Test sentence should produce errors"
        
        error = errors[0]
        assert hasattr(error, 'rule_type'), "Error missing rule_type"
        assert hasattr(error, 'message'), "Error missing message"
        assert hasattr(error, 'suggestion'), "Error missing suggestion"
        assert hasattr(error, 'context'), "Error missing context"
        assert hasattr(error, 'start_pos'), "Error missing start_pos"
        assert hasattr(error, 'end_pos'), "Error missing end_pos"

class TestAdverbRules:
    def test_adverb_placement(self, sample_doc):
        """Test adverb position detection"""
        errors = AdverbRules().apply(sample_doc)
        assert len(errors) == 1, "Should detect misplaced adverb"
        assert errors[0].rule_type == RuleType.ADVERB
        assert "souvent" in errors[0].context

    def test_compound_tense_adverbs(self):
        """Test adverb placement in compound tenses"""
        doc = FrenchGrammarRules().nlp("J'ai rapidement mangé.")
        errors = AdverbRules().apply(doc)
        assert len(errors) == 1, "Should flag adverb between auxiliary and past participle"

class TestAgreementRules:
    def test_adjective_agreement(self):
        """Test noun-adjective agreement"""
        doc = FrenchGrammarRules().nlp("La petit fille")
        errors = AgreementRules().apply(doc)
        assert len(errors) == 1, "Should detect gender disagreement"
        assert errors[0].rule_type == RuleType.AGREEMENT
        assert "petit" in errors[0].context

    def test_verb_agreement(self):
        """Test subject-verb agreement"""
        doc = FrenchGrammarRules().nlp("Les fille est belle")
        errors = AgreementRules().apply(doc)
        assert len(errors) >= 2, "Should detect number disagreement in both noun and verb"

class TestIntegration:
    def test_with_utils_module(self, grammar_rules):
        """Test integration with sentence analysis utils"""
        sentence = "Je souvent vais au parc"
        analysis = analyze_sentence_structure(sentence)
        errors = grammar_rules.check_sentence(sentence)
        
        # Verify adverb appears in both analysis and errors
        adverbs = [t['text'] for t in analysis if t['french_pos'] == 'Adverbe']
        assert "souvent" in adverbs, "Adverb not detected in analysis"
        
        error_adverbs = [e.context for e in errors if e.rule_type == RuleType.ADVERB]
        assert any("souvent" in ctx for ctx in error_adverbs), (
            "Adverb error not detected by rules"
        )

    def test_error_positions(self):
        """Test error position accuracy"""
        sentence = "La petit garçon mange rapidement."
        errors = FrenchGrammarRules().check_sentence(sentence)
        
        # Find the agreement error
        agreement_errors = [e for e in errors if e.rule_type == RuleType.AGREEMENT]
        assert agreement_errors, "No agreement error detected"
        
        error = agreement_errors[0]
        assert sentence[error.start_pos:error.end_pos] == "petit", (
            "Error position doesn't match the incorrect word"
        )