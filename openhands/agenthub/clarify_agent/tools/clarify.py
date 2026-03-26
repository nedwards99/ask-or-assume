# clarify.py
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from openhands.llm.tool_names import CLARIFY_TOOL_NAME

_CLARIFY_DESCRIPTION= """
This tool can be used to ask the user clarification questions when critical information is missing or ambiguous.

## Application Guidelines

Utilize this tool in the following situations:

1. Ambiguous requirements - Task has unclear scope, multiple interpretations, or missing specifications
2. Missing critical information - File paths, versions, configurations, constraints, or success criteria are undefined
3. Conflicting information - User request contradicts codebase reality, context, or previous instructions
4. Unclear success criteria - Multiple valid implementations exist without a clear preference

## Situations Where Tool Usage Is Unnecessary

Avoid using this tool when:

1. Tasks are simple and unambiguous with clear context (e.g., "Show me the contents of app.py")
2. Pure informational questions requiring no follow-up action
3. Safe defaults are obvious and well-documented
4. Standard conventions apply without user-specific requirements

## Best Practices

1. Question asking: Generate as many questions as needed to elicit the required information, but no more
2. Question format: Use closed-ended questions with explicit options when possible
3. Defaults: Propose a default option when it seems reasonable
4. Specificity: Reference exact files, line numbers, or values rather than vague descriptions

## Question Structure

Each question should contain:
- `id`: Stable identifier (e.g., "Q1", "config-target")
- `text`: Clear, specific question
- `options`: List of possible answers (when applicable)
- `default`: Your recommended option (when applicable)

## Usage Scenarios

**Scenario A: Ambiguous Test Request**
User request: "Run the tests and fix any failures"

Response approach: Before I run tests, I need clarification on a few points.

*Clarification questions:*
1. Which test suite should I run? (options: all, unit, integration; default: all)
2. What's your preferred fix approach? (options: fix code, update tests; default: fix code)
3. Is there anything else I should avoid changing?

**Scenario B: Mid-Task Discovery**
Context: Found 3 different config files during implementation

Response approach: I found 3 configuration files (dev.json, staging.json, prod.json). Before proceeding with updates, I need to clarify the target environment.

*Clarification question:*
1. Which environment should I update? (options: dev, staging, prod, all three; default: dev)

**Scenario C: Risky Operation**
User request: "Delete all the old log files"

Response approach: Before deleting files, I need to confirm the scope to prevent accidental data loss.

*Clarification questions:*
1. Which logs should I delete? (options: logs older than 30 days, logs older than 7 days, all logs; default: logs older than 30 days)
2. Should I create a backup first? (options: yes, no; default: yes)
"""

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
                        },
                        'required': ['id', 'text']
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

