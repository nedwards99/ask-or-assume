import asyncio
import json
import os

import pandas as pd
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from litellm import completion as litellm_completion

import openhands.agenthub
from evaluation.benchmarks.swe_bench.run_infer import (
    AgentFinishedCritic,
    complete_runtime,
    filter_dataset,
    get_config,
    initialize_runtime,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    get_metrics,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.utils.async_utils import call_async_from_sync

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'


class FakeUser:
    def __init__(self, issue, hints, files):
        self.system_message = f"""
        You are a GitHub user reporting an issue. Here are the details of your issue and environment:

        Issue: {issue}

        Hints: {hints}

        Files relative to your current directory: {files}

        Your task is to respond to questions from a coder who is trying to solve your issue. The coder has a summarized version of the issue you have. Follow these rules:
        1. If the coder asks a question that is directly related to the information in the issue you have, provide that information.
        2. Always stay in character as a user reporting an issue, not as an AI assistant.
        3. Keep your responses concise and to the point.
        4. The coder has limited turns to solve the issue. Do not interact with the coder beyond 3 turns.

        Respond with "I don't have that information" if the question is unrelated or you're unsure.
        """
        self.chat_history = [{'role': 'system', 'content': self.system_message}]
        self.turns = 0
        # Get LLM config from config.toml
        self.llm_config = get_llm_config_arg(
            'llm.fake_user'
        )  # You can change 'fake_user' to any config name you want

    def generate_reply(self, question):
        return ''

# Global variable for fake user
fake_user = None


def get_fake_user_response(state: State) -> str:
    return 'Please continue working on the task.'


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': get_fake_user_response,
}


# AGENT_CLS_TO_INST_SUFFIX = {
#     'CodeActAgent': 'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n',
#     'CodeActSWEAgent': 'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n',
# }


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    instruction = (
        '<uploaded_files>\n'
        f'/workspace/{workspace_dir_name}\n'
        '</uploaded_files>\n'
        f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        f'<pr_description>\n'
        f'{instance.original_issue}\n'
        '</pr_description>\n\n'
        '<hints>\n'
        f'{instance.hints_text}\n'
        '</hints>\n'
        'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        'Your task is to make the minimal changes to non-test files in the /repo directory to ensure the <pr_description> is satisfied.\n'
        'Follow these steps to resolve the issue:\n'
        '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
        '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error.\n'
        '3. Edit the source code of the repo to resolve the issue.\n'
        '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        '5. Think about edge cases and make sure your fix handles them as well.\n'
        "Your thinking should be thorough and so it's fine if it's very long.\n"
    )
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template('swe_default.j2')

    # Render swe_default.j2 but use original_issue as the issue text.
    instance_with_original_issue = instance.copy()
    instance_with_original_issue['problem_statement'] = instance.original_issue
    context = {
        'instance': instance_with_original_issue,
        'workspace_dir_name': workspace_dir_name,
        'metadata': metadata,
        'test_instructions': '',
    }
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\n'
            'You SHOULD NEVER attempt to browse the web. '
            '</IMPORTANT!>\n'
        )
    return MessageAction(content=instruction)



def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)
    global fake_user
    original_issue = instance.original_issue
    issue = str(original_issue)
    fake_user = FakeUser(issue=issue, hints=instance.hints_text, files=instance.files)
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance, metadata)

        message_action = get_instruction(instance, metadata)

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        if (
            state
            and state.last_error
            and 'fatal error during agent execution' in state.last_error
            and 'stuck in a loop' not in state.last_error
        ):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'git_patch': git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)
    # Save the output
    instruction = message_action.content
    if message_action.image_urls:
        instruction += (
            '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
        )
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='cmu-lti/interactive-swe',
        help='dataset to evaluate on',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )

    args, _ = parser.parse_known_args()

    # Load dataset from huggingface datasets
    dataset = load_dataset(args.dataset, split=args.split)
    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Get condenser config from environment variable
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        # If no specific condenser config is provided via env var, default to NoOpCondenser
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    details = {"mode": "full"}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        condenser_config=condenser_config,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)
    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=8
            * 60
            * 60,
        max_retries=5,
    )
