"""
Unit tests for French grammar rule implementations
Tests individual grammar rules in isolation
"""

import pytest
from spacy.tokens import Doc
from core.rules import (
    FrenchGrammarRules,
    GrammarError,
    RuleType,
    AdverbRules,
    AgreementRules,
    ConjugationRules,
    SentenceStructureRules
)
from app.common.utils import get_french_pos_tag

# Test sentences with expected results
ADVERB_TEST_CASES = [
    ("Je mange rapidement.", True, None),  # Correct
    ("Je rapidement mange.", False, "adverb_placement"),  # Wrong position
    ("J'ai rapidement mangé.", False, "compound_tense_adverb"),  # Wrong in compound tense
]

AGREEMENT_TEST_CASES = [
    ("La belle fille", True, None),
    ("Le belle fille", False, "gender_agreement"),
    ("Les fille", False, "number_agreement"),
    ("Ils mange", False, "subject_verb_agreement"),
]

CONJUGATION_TEST_CASES = [
    ("Je vais au marché", True, None),
    ("Je va au marché", False, "present_conjugation"),
    ("Nous sommes allé", False, "past_participle"),
]

STRUCTURE_TEST_CASES = [
    ("Le chat noir dort.", True, None),
    ("Noir le chat dort.", False, "adjective_placement"),
    ("Dort le chat noir.", False, "verb_placement"),
]

@pytest.fixture
def nlp():
    return FrenchGrammarRules().nlp

@pytest.fixture
def adverb_rules():
    return AdverbRules()

@pytest.fixture
def agreement_rules():
    return AgreementRules()

@pytest.fixture
def conjugation_rules():
    return ConjugationRules()

@pytest.fixture
def structure_rules():
    return SentenceStructureRules()

class TestAdverbRules:
    @pytest.mark.parametrize("sentence,should_pass,error_type", ADVERB_TEST_CASES)
    def test_adverb_placement(self, nlp, adverb_rules, sentence, should_pass, error_type):
        doc = nlp(sentence)
        errors = adverb_rules.apply(doc)
        
        if should_pass:
            assert not errors, f"Expected no errors but got: {errors}"
        else:
            assert errors, "Expected adverb placement error"
            assert any(e.rule_type == RuleType.ADVERB for e in errors), (
                "Expected adverb-related error"
            )
            
            if error_type == "compound_tense_adverb":
                assert any("compound tense" in e.message.lower() for e in errors), (
                    "Should mention compound tense in error"
                )

    def test_adverb_after_auxiliary(self, nlp, adverb_rules):
        doc = nlp("J'ai rapidement mangé.")
        errors = adverb_rules.apply(doc)
        assert len(errors) == 1
        assert "after the auxiliary" in errors[0].suggestion.lower()

class TestAgreementRules:
    @pytest.mark.parametrize("sentence,should_pass,error_type", AGREEMENT_TEST_CASES)
    def test_agreements(self, nlp, agreement_rules, sentence, should_pass, error_type):
        doc = nlp(sentence)
        errors = agreement_rules.apply(doc)
        
        if should_pass:
            assert not errors, f"Expected no errors but got: {errors}"
        else:
            assert errors, f"Expected {error_type} error"
            assert any(e.rule_type == RuleType.AGREEMENT for e in errors)
            
            if error_type == "gender_agreement":
                assert any("gender" in e.message.lower() for e in errors)
            elif error_type == "number_agreement":
                assert any("number" in e.message.lower() or "plural" in e.message.lower() for e in errors)

    def test_verb_agreement_details(self, nlp, agreement_rules):
        doc = nlp("Les fille mange.")
        errors = agreement_rules.apply(doc)
        
        # Should catch both noun and verb agreement errors
        assert len(errors) >= 2
        noun_errors = [e for e in errors if "fille" in e.context]
        verb_errors = [e for e in errors if "mange" in e.context]
        assert noun_errors and verb_errors

class TestConjugationRules:
    @pytest.mark.parametrize("sentence,should_pass,error_type", CONJUGATION_TEST_CASES)
    def test_conjugations(self, nlp, conjugation_rules, sentence, should_pass, error_type):
        doc = nlp(sentence)
        errors = conjugation_rules.apply(doc)
        
        if should_pass:
            assert not errors, f"Expected no errors but got: {errors}"
        else:
            assert errors, f"Expected {error_type} error"
            assert any(e.rule_type == RuleType.CONJUGATION for e in errors)
            
            if error_type == "present_conjugation":
                assert any("conjugation" in e.message.lower() for e in errors)
            elif error_type == "past_participle":
                assert any("participle" in e.message.lower() for e in errors)

    def test_compound_tense_conjugation(self, nlp, conjugation_rules):
        doc = nlp("Nous avons mangéons.")
        errors = conjugation_rules.apply(doc)
        assert len(errors) == 1
        assert "past participle" in errors[0].message.lower()

class TestSentenceStructureRules:
    @pytest.mark.parametrize("sentence,should_pass,error_type", STRUCTURE_TEST_CASES)
    def test_structure(self, nlp, structure_rules, sentence, should_pass, error_type):
        doc = nlp(sentence)
        errors = structure_rules.apply(doc)
        
        if should_pass:
            assert not errors, f"Expected no errors but got: {errors}"
        else:
            assert errors, f"Expected {error_type} error"
            assert any(e.rule_type == RuleType.STRUCTURE for e in errors)
            
            if error_type == "adjective_placement":
                assert any("adjective" in e.message.lower() for e in errors)
            elif error_type == "verb_placement":
                assert any("verb position" in e.message.lower() for e in errors)

    def test_subject_verb_order(self, nlp, structure_rules):
        doc = nlp("Mange le chat.")
        errors = structure_rules.apply(doc)
        assert len(errors) == 1
        assert "subject-verb" in errors[0].message.lower()

class TestErrorFormatting:
    """Test the GrammarError formatting and positions"""
    
    def test_error_positions(self, nlp, agreement_rules):
        sentence = "La petit garçon"
        doc = nlp(sentence)
        errors = agreement_rules.apply(doc)
        
        assert len(errors) == 1
        error = errors[0]
        
        # Verify positions match the actual word
        assert error.start_pos == sentence.find("petit")
        assert error.end_pos == error.start_pos + len("petit")
        
        # Verify context contains surrounding words
        assert "La petit garçon" in error.context
        
        # Verify suggestion is helpful
        assert "petite" in error.suggestion.lower()

    def test_multiple_errors(self, nlp):
        sentence = "Le fille va au parc et mange une pomme vert."
        errors = FrenchGrammarRules().check_sentence(sentence)
        
        # Should catch at least gender and number agreement errors
        assert len(errors) >= 2
        assert any(e.rule_type == RuleType.AGREEMENT for e in errors)
        assert any("fille" in e.context for e in errors)
        assert any("vert" in e.context for e in errors)