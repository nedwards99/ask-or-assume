from __future__ import annotations

"""
Agent controller variant with lightweight delegate pooling.

Key differences vs. AgentController:
- Reuse a delegate controller per agent name instead of creating/closing one
  for every AgentDelegateAction.
- Do not close delegates on end_delegate; park them for reuse and close them
  when the parent controller closes.
"""

import inspect
from typing import Any

from openhands.controller import agent_controller as base
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import EventSource
from openhands.events.action import AgentDelegateAction
from openhands.events.observation import AgentDelegateObservation


class AgentControllerV2(base.AgentController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache delegate controllers by agent name for reuse.
        self._delegate_pool: dict[str, base.AgentController] = {}

    async def start_delegate(self, action: AgentDelegateAction) -> None:
        """Start or reuse a delegate agent."""
        # Reuse a parked delegate if available
        delegate_ctrl = self._delegate_pool.get(action.agent)
        if delegate_ctrl is not None:
            # Re-link shared flags/metrics to the parent before reuse
            delegate_ctrl.state.iteration_flag = self.state.iteration_flag
            delegate_ctrl.state.budget_flag = self.state.budget_flag
            delegate_ctrl.state.metrics = self.state.metrics
            delegate_ctrl.state.parent_metrics_snapshot = (
                self.state_tracker.get_metrics_snapshot()
            )
            delegate_ctrl.state.parent_iteration = (
                self.state.iteration_flag.current_value
            )
            delegate_ctrl.state.delegate_level = self.state.delegate_level + 1
            delegate_ctrl.state_tracker.state = delegate_ctrl.state

            # Share history with parent
            delegate_ctrl.state.history = self.state.history
            delegate_ctrl.state.start_id = self.state.start_id

            self.delegate = delegate_ctrl
            await self.delegate.set_agent_state_to(AgentState.RUNNING)
            return

        agent_cls: type[Agent] = Agent.get_cls(action.agent)
        agent_config = self.agent_configs.get(action.agent, self.agent.config)
        shared_kwargs: dict[str, Any] = {}
        if hasattr(self.agent, 'condenser'):
            try:
                init_params = inspect.signature(agent_cls.__init__).parameters
            except (TypeError, ValueError):
                init_params = {}
            if 'shared_condenser' in init_params:
                shared_kwargs['shared_condenser'] = getattr(self.agent, 'condenser')

        delegate_agent = agent_cls(
            config=agent_config,
            llm_registry=self.agent.llm_registry,
            **shared_kwargs,
        )

        state = State(
            session_id=self.id.removesuffix('-delegate'),
            user_id=self.user_id,
            inputs=action.inputs or {},
            iteration_flag=self.state.iteration_flag,
            budget_flag=self.state.budget_flag,
            delegate_level=self.state.delegate_level + 1,
            metrics=self.state.metrics,
            start_id=self.state.start_id,
            parent_metrics_snapshot=self.state_tracker.get_metrics_snapshot(),
            parent_iteration=self.state.iteration_flag.current_value,
        )
        self.log(
            'debug',
            f'start delegate, creating agent {delegate_agent.name}',
        )

        self.delegate = AgentControllerV2(
            sid=self.id + '-delegate',
            file_store=self.file_store,
            user_id=self.user_id,
            agent=delegate_agent,
            event_stream=self.event_stream,
            conversation_stats=self.conversation_stats,
            iteration_delta=self._initial_max_iterations,
            budget_per_task_delta=self._initial_max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            initial_state=state,
            is_delegate=True,
            headless_mode=self.headless_mode,
            security_analyzer=self.security_analyzer,
        )
        self._delegate_pool[action.agent] = self.delegate

    def end_delegate(self) -> None:
        """Park the delegate for reuse, emit its result, and resume parent."""
        if self.delegate is None:
            return

        delegate_state = self.delegate.get_agent_state()
        self.state.iteration_flag.current_value = (
            self.delegate.state.iteration_flag.current_value
        )

        delegate_metrics = self.state.get_local_metrics()
        logger.info(f'Local metrics for delegate: {delegate_metrics}')

        if delegate_state in (AgentState.FINISHED, AgentState.REJECTED):
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )
            display_outputs = {k: v for k, v in delegate_outputs.items()}
            formatted_output = ', '.join(
                f'{key}: {value}' for key, value in display_outputs.items()
            )
            content = (
                f'{self.delegate.agent.name} finishes task with {formatted_output}'
            )
        else:
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )
            content = (
                f'{self.delegate.agent.name} encountered an error during execution.'
            )

        content = f'Delegated agent finished with result:\n\n{content}'

        obs = AgentDelegateObservation(outputs=delegate_outputs, content=content)
        delegate_action_id = getattr(self._pending_action, 'id', None)
        obs._cause = delegate_action_id

        for event in reversed(self.state.history):
            if isinstance(event, AgentDelegateAction):
                delegate_action = event
                obs.tool_call_metadata = delegate_action.tool_call_metadata
                break

        self.event_stream.add_event(obs, EventSource.AGENT)
        # park delegate; keep it in the pool for reuse
        self.delegate = None

    async def close(self, set_stop_state: bool = True) -> None:
        """Close parent and any pooled delegates."""
        await super().close(set_stop_state=set_stop_state)
        for delegate in self._delegate_pool.values():
            if not getattr(delegate, '_closed', False):
                await delegate.close(set_stop_state=set_stop_state)
