# API endpoints

from fastapi import APIRouter
from app.core.grammar_checker import FrenchGrammarChecker

router = APIRouter()
checker = FrenchGrammarChecker()

@router.post("/check")
async def check_grammar(text: str):
    return checker.analyze_sentence(text)