# [MODULE DOC]
# Unit tests for PPOTrainer._requires_proxy_workflow (decides whether a workflow given
# as instance, class or import string is an agent-style workflow needing the OpenAI
# proxy) and for the DeprecationWarning emitted when AgentWorkflow subclasses are
# instantiated.
"""Unit tests for workflow detection logic in PPOTrainer."""

import warnings

import pytest

from areal.api import AgentWorkflow, RolloutWorkflow


class DummyRolloutWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # HELPER: Trivial RolloutWorkflow subclass for detection tests.
    """Test RolloutWorkflow implementation."""

    async def arun_episode(self, engine, data):
        # [FUNCTION DOC]
        # Returns an empty dict; ignores engine and data.
        return {}


class DummyAgentNonInheriting:
    # [CLASS DOC]
    # HELPER: Duck-typed agent with an async run() but no base class.
    """No inheritance - duck typing only."""

    async def run(self, data, **extra_kwargs):
        # [FUNCTION DOC]
        # Returns reward 1.0; ignores inputs.
        return 1.0


class TestWorkflowDetection:
    # [CLASS DOC]
    # Detection tests over instances, classes and string paths.
    """Test the _requires_proxy_workflow detection logic."""

    @pytest.fixture
    def trainer_with_detection(self):
        # [FUNCTION DOC]
        # Fixture: returns an object whose _requires_proxy_workflow delegates to the unbound
        # PPOTrainer method (avoids constructing a full trainer).
        """Create a minimal object with the detection method."""
        # We can't easily instantiate a full PPOTrainer in unit tests,
        # so we'll create a simple object with just the method we need
        from areal.trainer import PPOTrainer

        class MinimalTrainer:
            # [CLASS DOC]
            # HELPER: Shell object carrying only the detection method.
            def _requires_proxy_workflow(self, workflow):
                # [FUNCTION DOC]
                # Calls PPOTrainer._requires_proxy_workflow with self and returns its bool.
                # Copy the implementation from PPOTrainer
                return PPOTrainer._requires_proxy_workflow(self, workflow)

        return MinimalTrainer()

    @pytest.fixture
    def dummy_agent_workflow_class(self):
        # [FUNCTION DOC]
        # Fixture: defines and returns a deprecated AgentWorkflow subclass lazily.
        """Create a DummyAgentWorkflow class inside fixture to avoid module-level warning."""

        class DummyAgentWorkflow(AgentWorkflow):
            # [CLASS DOC]
            # HELPER: AgentWorkflow subclass whose run returns 1.0.
            """Test AgentWorkflow implementation (deprecated)."""

            async def run(self, data, **extra_kwargs):
                # [FUNCTION DOC]
                # Returns 1.0.
                return 1.0

        return DummyAgentWorkflow

    def test_rollout_workflow_instance_no_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts a RolloutWorkflow instance does not require the proxy.
        """RolloutWorkflow instances should NOT require proxy."""
        workflow = DummyRolloutWorkflow()
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_rollout_workflow_class_no_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts a RolloutWorkflow class does not require the proxy.
        """RolloutWorkflow classes should NOT require proxy."""
        assert not trainer_with_detection._requires_proxy_workflow(DummyRolloutWorkflow)

    def test_rollout_workflow_string_no_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts the RLVRWorkflow import string does not require the proxy.
        """String paths to RolloutWorkflow should NOT require proxy."""
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_agent_workflow_instance_requires_proxy(
        self, trainer_with_detection, dummy_agent_workflow_class
    ):
        # [FUNCTION DOC]
        # Asserts an AgentWorkflow instance requires the proxy and that exactly one
        # DeprecationWarning mentioning the deprecated base was emitted.
        """AgentWorkflow instances should require proxy and emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            workflow = dummy_agent_workflow_class()
            assert trainer_with_detection._requires_proxy_workflow(workflow)
            # Verify deprecation warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "inherits from deprecated AgentWorkflow" in str(w[0].message)

    def test_agent_workflow_class_requires_proxy(
        self, trainer_with_detection, dummy_agent_workflow_class
    ):
        # [FUNCTION DOC]
        # Asserts an AgentWorkflow subclass (class object) requires the proxy.
        """AgentWorkflow classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(
            dummy_agent_workflow_class
        )

    def test_non_inheriting_agent_instance_requires_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts a duck-typed agent instance requires the proxy.
        """Non-inheriting agent instances should require proxy."""
        workflow = DummyAgentNonInheriting()
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_non_inheriting_agent_class_requires_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts a duck-typed agent class requires the proxy.
        """Non-inheriting agent classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(DummyAgentNonInheriting)

    def test_invalid_string_requires_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts an unimportable string path is treated as requiring the proxy (fail-safe).
        """Invalid string paths should require proxy (fail-safe)."""
        workflow = "nonexistent.module.Workflow"
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_string_to_agent_workflow_requires_proxy(self, trainer_with_detection):
        # [FUNCTION DOC]
        # Asserts a string path to a duck-typed test agent requires the proxy.
        """String paths to agent workflows should require proxy.

        Note: SimpleAgent is a duck-typed agent (no AgentWorkflow inheritance),
        demonstrating that any class with a compatible run() method works.
        """
        workflow = "tests.experimental.openai.utils.SimpleAgent"
        assert trainer_with_detection._requires_proxy_workflow(workflow)


class TestAgentWorkflowDeprecation:
    # [CLASS DOC]
    # Tests of the AgentWorkflow deprecation warning.
    """Test that AgentWorkflow deprecation warning works correctly."""

    def test_deprecation_warning_on_instantiation(self):
        # [FUNCTION DOC]
        # Asserts instantiating an AgentWorkflow subclass emits one DeprecationWarning naming it.
        """Verify deprecation warning is triggered when instantiating AgentWorkflow subclass."""

        class MyAgent(AgentWorkflow):
            # [CLASS DOC]
            # HELPER: Local AgentWorkflow subclass.
            async def run(self, data, **extra_kwargs):
                # [FUNCTION DOC]
                # Returns 1.0.
                return 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _agent = MyAgent()  # noqa: F841
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MyAgent inherits from deprecated AgentWorkflow" in str(w[0].message)

    def test_deprecation_warning_without_super_init(self):
        # [FUNCTION DOC]
        # Asserts the warning is emitted (via metaclass) even if __init__ skips super(), and
        # that the subclass __init__ still ran.
        """Verify warning triggers even if subclass doesn't call super().__init__()."""

        class MyAgentNoSuper(AgentWorkflow):
            # [CLASS DOC]
            # HELPER: AgentWorkflow subclass whose __init__ does not call super().
            def __init__(self):
                # [FUNCTION DOC]
                # Sets custom_attr = "test" without calling the base initializer.
                # Intentionally not calling super().__init__()
                self.custom_attr = "test"

            async def run(self, data, **extra_kwargs):
                # [FUNCTION DOC]
                # Returns 1.0.
                return 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = MyAgentNoSuper()
            # Warning should still be triggered via metaclass
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MyAgentNoSuper inherits from deprecated AgentWorkflow" in str(
                w[0].message
            )
            # Verify the instance was created correctly
            assert agent.custom_attr == "test"
