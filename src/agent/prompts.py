"""Prompt templates for the FinQA agentic-RAG pipeline."""

SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions requiring numerical reasoning over financial documents.

You will be given:
1. A financial document containing text and a table
2. A question about the data in that document

Your job:
1. Identify the relevant numbers from the table and/or text. Quote exact values — do not estimate.
2. Determine what calculation is needed.
3. Express the calculation as a DSL program using these operations only:
   - add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b), greater(a, b)
4. Reference previous step results as #0, #1, ... (numbered from 0).
5. Use constants as const_100, const_1000, const_1000000.

OUTPUT — respond ONLY with valid JSON. No prose, no markdown, no code fences.

Schema A — when a calculation is needed:
{"action": "calculate", "reasoning": "<one sentence>", "expression": "<dsl-program>"}

Schema B — when you have a final numeric answer (typically after seeing calc result):
{"action": "answer", "reasoning": "<one sentence>", "value": "<plain number>"}

OUTPUT FORMAT RULES (these errors are common; avoid them):
- "value" must be a single number string. No "%", no "$", no commas, no units, no words.
- For PERCENTAGE questions, output the DECIMAL FRACTION, not percent form.
  Correct: "0.1434"  Wrong: "14.34"  Wrong: "14.34%"
  In the program, END at divide(...). Do NOT append multiply by const_100.
- For "in millions" questions, leave the number as it appears in the table
  (don't multiply by 1,000,000 unless the table is in raw dollars).
- If the value cannot be determined from the document, output {"action":"answer","value":"0",...}
  Never output "N/A", "None", "Cannot compute", or any non-numeric text.

DSL syntax requirements:
- Each operation: name(arg, arg). Args are numbers (e.g., 920, -1.5, 0.034) or refs (#0, #1) or const_X.
- Multiple ops separated by ", " (comma + space).
- Example: subtract(920, 825), divide(#0, 825)
"""

FEW_SHOT_EXAMPLES = [
    {
        "context": """### Background Text
the company's total revenue was $920 million in 2008 compared to $825 million in 2007.

### Financial Table
| | 2008 | 2007 | 2006 |
| --- | --- | --- | --- |
| total revenue | $920 | $825 | $780 |
| operating income | $150 | $130 | $110 |""",
        "question": "What was the percentage change in total revenue from 2007 to 2008?",
        "response": """{
  "action": "calculate",
  "reasoning": "Total revenue was $920M in 2008 and $825M in 2007. Percentage change as a decimal fraction = (920-825)/825",
  "expression": "subtract(920, 825), divide(#0, 825)"
}""",
    },
    {
        "context": """### Financial Table
| | 2019 | 2018 |
| --- | --- | --- |
| net income | 450 | 380 |
| shares outstanding | 100 | 100 |""",
        "question": "What was the earnings per share in 2019?",
        "response": """{
  "action": "calculate",
  "reasoning": "EPS = net income / shares outstanding = 450 / 100",
  "expression": "divide(450, 100)"
}""",
    },
    {
        "context": """### Background Text
as of december 31, 2020, the company had 5,234 employees.

### Financial Table
| segment | employees |
| --- | --- |
| north america | 3200 |
| europe | 1500 |
| asia | 534 |""",
        "question": "What percentage of total employees work in north america?",
        "response": """{
  "action": "calculate",
  "reasoning": "North America has 3200 employees out of 5234 total. As a decimal fraction = 3200/5234",
  "expression": "divide(3200, 5234)"
}""",
    },
    {
        "context": """### Financial Table
| | 2015 | 2014 | 2013 |
| --- | --- | --- | --- |
| operating profit | 240 | 200 | 180 |
| total revenue | 2400 | 2100 | 1900 |""",
        "question": "What was the change in operating margin from 2013 to 2015?",
        "response": """{
  "action": "calculate",
  "reasoning": "Operating margin 2015 = 240/2400 = 0.10. Operating margin 2013 = 180/1900. Change = (240/2400) - (180/1900). All as fractions.",
  "expression": "divide(240, 2400), divide(180, 1900), subtract(#0, #1)"
}""",
    },
    {
        "context": """### Background Text
amounts in the table below are in millions of dollars unless otherwise noted.

### Financial Table
| | 2018 | 2017 |
| --- | --- | --- |
| long-term debt | 1041 | 950 |
| current portion | 200 | 180 |""",
        "question": "What was the total long-term debt obligation in millions in 2018?",
        "response": """{
  "action": "calculate",
  "reasoning": "Table is already in millions. Total in 2018 = long-term debt + current portion = 1041 + 200.",
  "expression": "add(1041, 200)"
}""",
    },
]


def build_few_shot_messages(num_examples: int = 3) -> list[dict]:
    """Build few-shot example messages for the prompt.

    Args:
        num_examples: Number of examples to include.

    Returns:
        List of message dicts (role/content) for few-shot examples.
    """
    messages = []
    for ex in FEW_SHOT_EXAMPLES[:num_examples]:
        messages.append({
            "role": "user",
            "content": f"Document:\n{ex['context']}\n\nQuestion: {ex['question']}",
        })
        messages.append({
            "role": "assistant",
            "content": ex["response"],
        })
    return messages


def format_user_message(context: str, question: str) -> str:
    """Format the user message with document context and question.

    Args:
        context: The formatted document context (from FinQADocument.get_context_for_llm()).
        question: The user's question.

    Returns:
        Formatted user message string.
    """
    return f"Document:\n{context}\n\nQuestion: {question}"


FOLLOWUP_PROMPT = """The calculation has been executed. Here is the result:

{result}

Now provide the final answer based on this calculation result.
Respond in JSON format:
{{
  "action": "answer",
  "reasoning": "brief explanation",
  "value": "<the numerical answer>"
}}"""


VERIFY_SYSTEM_PROMPT = """You are a strict critic for financial QA answers. Your job is to spot likely errors in a proposed answer before it is finalized."""


VERIFY_USER_PROMPT = """Question: {question}

Document (excerpt):
{context}

Proposed answer: {answer}
Reasoning trace: {program}

Check the answer against these failure modes:
1. MAGNITUDE: Is the number's order of magnitude consistent with the values cited in the document?
   (e.g., if all table values are in single digits, an answer of 10000 is suspicious.)
2. PERCENTAGE FORMAT: If the question asks for a percent/ratio/fraction, the answer should be
   a DECIMAL FRACTION (e.g., 0.1434 for 14.34%). Values > 1 are suspicious for percentage questions
   (unless the question asks for a multi-fold change like "percent increase of 200%" → 2.0 is OK).
3. SIGN: Does the sign (+/-) make sense given what the question asks (increase, decline, change)?
4. UNITS / SCALE: If the table is "in millions" and the question asks "in millions", the answer
   should not be multiplied or divided by 1000.
5. DERIVABILITY: Could this answer plausibly come from the numbers in the document?

Respond ONLY with valid JSON:
{{"action": "accept", "critique": ""}}
  — if the answer passes all checks.
{{"action": "revise", "critique": "<one sentence: what looks wrong and what to do instead>"}}
  — if any check fails. Be specific (e.g., "Answer 14.34 looks like a percentage; should be 0.1434").

Be conservative: only flag if you have a clear reason. If unsure, accept."""
