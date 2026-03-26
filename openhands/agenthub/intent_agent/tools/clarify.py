# from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# from openhands.llm.tool_names import CLARIFY_TOOL_NAME

# _CLARIFY_DESCRIPTION = """Ask the user a question or request clarification.

# Use this tool when more context, confirmation, or additional details from the user are required before proceeding."""

# ClarifyTool = ChatCompletionToolParam(
#     type='function',
#     function=ChatCompletionToolParamFunctionChunk(
#         name=CLARIFY_TOOL_NAME,
#         description=_CLARIFY_DESCRIPTION,
#         parameters={
#             'type': 'object',
#             'properties': {
#                 'message': {
#                     'type': 'string',
#                     'description': 'The question or clarification request to send to the user.',
#                 },
#                 'wait_for_response': {
#                     'type': 'boolean',
#                     'description': 'Whether to wait for the user to respond before taking further actions. Defaults to true.',
#                 },
#             },
#             'required': ['message'],
#             'additionalProperties': False,
#         },
#     ),
# )

# clarify.py
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from openhands.llm.tool_names import CLARIFY_TOOL_NAME

_CLARIFY_DESCRIPTION= """
This tool can be used to ask the user batched clarification questions when critical information is missing or ambiguous.

## Application Guidelines

Utilize this tool in the following situations:

1. Ambiguous requirements - Task has unclear scope, multiple interpretations, or missing specifications
2. Missing critical information - File paths, versions, configurations, constraints, or success criteria are undefined
3. Risky or irreversible operations - Deletions, deployments, mass edits, or destructive changes
4. Conflicting information - User request contradicts codebase reality, context, or previous instructions
5. Unclear success criteria - Multiple valid implementations exist without clear preference

## Situations Where Tool Usage Is Unnecessary

Avoid using this tool when:

1. Tasks are simple and unambiguous with clear context
2. Pure informational questions requiring no follow-up action
3. Safe defaults are obvious and well-documented
4. Standard conventions apply without user-specific requirements

## Usage Practices

1. **Question Batching**: Ask 2-5 related questions per call
2. **Question Format**: Use closed-ended questions with explicit options when possible
3. **Defaults**: Always propose a recommended default for each unclear item
4. **Specificity**: Reference exact files, line numbers, or values rather than vague descriptions
5. **Urgency**: Set `required: true` only for questions that must block execution

## Question Structure

Each question should contain:
- `id`: Stable identifier (e.g., "Q1", "config-target")
- `text`: Clear, specific question
- `options`: List of possible answers (when applicable)
- `default`: Your recommended option
- `required`: Whether execution must wait for answer (default: true)

## Usage Scenarios

**Scenario A: Ambiguous Test Request**
User request: "Run the tests and fix any failures"

Response approach: Before I run tests, I need clarification on a few points.

*Clarification questions:*
1. Which test suite should I run? (options: all tests, unit only, integration only; default: all tests)
2. What's your preferred fix approach? (options: fix code to match tests, update tests to match code; default: fix code to match tests)
3. Should I commit fixes automatically? (options: yes, no show me first; default: no show me first)

**Scenario B: Mid-Task Discovery**
Context: Found 3 different config files during implementation

Response approach: I found 3 configuration files (dev.json, staging.json, prod.json). Before proceeding with updates, I need to clarify the target environment.

*Clarification question:*
1. Which environment should I update? (options: dev only, staging only, prod only, all three; default: dev only; required: true)

**Scenario C: Risky Operation**
User request: "Delete all the old log files"

Response approach: Before deleting files, I need to confirm the scope to prevent accidental data loss.

*Clarification questions:*
1. Which logs should I delete? (options: logs older than 30 days, logs older than 7 days, all logs; default: logs older than 30 days)
2. Should I create a backup first? (options: yes, no; default: yes)

## Counter-examples - Direct Implementation Preferred

**Counter-example A: Clear Simple Task**
User request: "Show me the contents of app.py"

Response approach: *Directly run `cat app.py` without clarification*

**Counter-example B: Standard Convention**
User request: "Add a README to the project"

Response approach: *Create README.md following standard conventions without clarification*

## Optional: Requirements Checklist

You may optionally include a requirements checklist to communicate your understanding:
```json
{
  "checklist": [
    {"id": "C1", "label": "Test command", "status": "UNKNOWN", "value": "pytest -q (proposed)"},
    {"id": "C2", "label": "Target modules", "status": "OK", "value": "core, api"}
  ]
}"""

# _CLARIFY_DESCRIPTION = """
# This tool sends a batched set of clarification questions to the user and (optionally) echoes a requirements checklist snapshot for confirmation.

# ## Application Guidelines
# Use this tool whenever critical requirements are UNKNOWN or MISSING, or when assumptions would materially affect correctness, safety, scope, or acceptance criteria.

# Typical triggers:
# 1. Ambiguous inputs (files, paths, versions, environments)
# 2. Unclear constraints (time/budget limits, performance targets)
# 3. Risky/irreversible steps (data deletion, large edits)
# 4. Multiple interpretations of scope or success criteria
# 5. Conflicting context or tool output that contradicts assumptions

# ## Usage
# - Batch 2-5 questions in one call. Prefer closed-ended questions with options.
# - Propose safe defaults for each unclear item and request approval.
# - If you maintain a requirements checklist, include a concise snapshot for user confirmation.

# ## Status values:
#   - OK: Requirement is applicable and its value/decision is confirmed.
#   - UNKNOWN: Requirement is applicable, but its specific value/decision is not yet known.
#     - *Example:* Output format is required → value (JSON vs Markdown) not decided.
#   - MISSING: Applicability or structure of the requirement is not yet established.
#     - *Example:* We don't know if there's a deployment target at all → confirm existence first.
#   - N/A: Confirmed not applicable (only set after explicit confirmation).

# ## Situations Where Tool Usage Is Unnecessary
# - Pure fact Q&A that needs no follow-up action
# - Trivial tasks where assumptions are clearly safe and documented already

# ## Example
# User: “Run tests and fix any failures.”
# Assistant (clarify):
# - message: "Before I run tests, I need a couple of confirmations."
# - checklist: [{id:"CHK-1",label:"Test command",status:"UNKNOWN",value:"pytest -q (proposed)"}]
# - questions: [
#     {id:"Q1",text:"Run tests from repo root?",options:["yes","no"],default:"yes"},
#     {id:"Q2",text:"May I modify core and api modules?",options:["core","api","both","neither"],default:"both",required:true}
#   ]
# - wait_for_response: true
# """

# _CLARIFY_DESCRIPTION = """Ask the user a batched set of clarification questions.
# Use this tool whenever key requirements are UNKNOWN or MISSING before proceeding."""

ClarifyTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CLARIFY_TOOL_NAME,
        description=_CLARIFY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                # Optional preamble/summary
                'message': {
                    'type': 'string',
                    'description': 'Optional short preface giving context for the questions.',
                },
                # Primary: batched questions
                'questions': {
                    'type': 'array',
                    'minItems': 1,
                    'description': 'Batched questions to the user.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string', 'description': 'Stable identifier (e.g., CHK-1).'},
                            'text': {'type': 'string', 'description': 'The question to ask.'},
                            'options': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Optional multiple-choice options.'
                            },
                            'default': {'type': 'string', 'description': 'Proposed default answer (if any).'},
                            'required': {'type': 'boolean', 'description': 'If true, block execution until answered.', 'default': True}
                        },
                        'required': ['id', 'text']
                    }
                },
                # Optional: echo the derived checklist snapshot
                'checklist': {
                    'type': 'array',
                    'description': 'Optional derived requirements checklist for the user to confirm.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'label': {'type': 'string'},
                            'status': {'type': 'string', 'enum': ['OK','UNKNOWN','MISSING']},
                            'value': {'type': 'string'}
                        },
                        'required': ['id', 'label', 'status']
                    }
                },
                'wait_for_response': {
                    'type': 'boolean',
                    'description': 'Whether to wait for user response before further actions.',
                    'default': True,
                },
            },
            'required': ['questions'],
            'additionalProperties': False,
        },
    ),
)

