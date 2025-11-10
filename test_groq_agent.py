

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

# Import components to test
from groq_colab_agent_complete import (
    ConfigManager,
    StorageManager,
    MeteringService,
    GroqColabAgent,
    Dashboard,
    VisualAnalytics,
    MeteringData,
    UsageSummary,
    OperationType,
    TaskStatus,
    TaskPriority
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def config():
    """Create test configuration"""
    return ConfigManager()


@pytest.fixture
def storage(temp_db):
    """Create test storage"""
    return StorageManager(db_path=temp_db)


@pytest.fixture
def metering_service(storage, config):
    """Create metering service"""
    return MeteringService(storage, config)


@pytest.fixture
async def agent(config):
    """Create test agent"""
    return GroqColabAgent(
        api_key="test-key",
        model="mixtral-8x7b-32768",
        config=config
    )


@pytest.fixture
def sample_task():
    """Create sample task"""
    return {
        "actionboard_id": "board-123",
        "prompt": "Test prompt",
        "action_type": "email_analysis",
        "context": "Test context"
    }


@pytest.fixture
def sample_metering_data():
    """Create sample metering data"""
    return MeteringData(
        action_id="action-001",
        actionboard_id="board-123",
        credits_used=5,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        operation_type="email_analysis",
        timestamp=datetime.now().isoformat(),
        status="completed",
        processing_time_ms=45.5
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigManager:
    """Test configuration management"""
    
    def test_config_initialization(self, config):
        """Test config loads correctly"""
        assert config is not None
        assert config.get("agent_name") == "GroqColab"
        assert config.get("credit_limit") == 1000
    
    def test_config_get_method(self, config):
        """Test get method with defaults"""
        assert config.get("groq_model") == "mixtral-8x7b-32768"
        assert config.get("nonexistent", "default") == "default"
    
    def test_config_all_required_keys(self, config):
        """Test all required config keys present"""
        required_keys = [
            "groq_api_key", "groq_model", "agent_name",
            "credit_limit", "token_limit", "storage_type"
        ]
        for key in required_keys:
            assert config.get(key) is not None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: STORAGE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestStorageManager:
    """Test storage management"""
    
    def test_db_initialization(self, storage):
        """Test database initializes"""
        # Check if tables exist by querying
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        required_tables = [
            "metering_transactions",
            "usage_tracking",
            "actionboard_limits",
            "task_tracking"
        ]
        for table in required_tables:
            assert table in tables
    
    def test_insert_metering(self, storage, sample_metering_data):
        """Test inserting metering data"""
        result = storage.insert_metering(sample_metering_data)
        assert result is True
    
    def test_get_usage_summary(self, storage):
        """Test retrieving usage summary"""
        result = storage.get_usage_summary("board-123")
        # Could be None if not initialized
        assert result is None or isinstance(result, dict)
    
    def test_get_billing_data(self, storage, sample_metering_data):
        """Test retrieving billing data"""
        storage.insert_metering(sample_metering_data)
        result = storage.get_billing_data("board-123", days=30)
        
        assert isinstance(result, list)
        if result:
            assert "action_id" in result[0]
            assert "credits_used" in result[0]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: METERING SERVICE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMeteringService:
    """Test metering service"""
    
    @pytest.mark.asyncio
    async def test_check_credits(self, metering_service):
        """Test credit checking"""
        result = await metering_service.check_credits("board-123", 100)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_consume_credits(self, metering_service):
        """Test credit consumption"""
        credits = await metering_service.consume_credits(
            "board-123", "email_analysis"
        )
        assert credits == 5  # Default for email_analysis
    
    @pytest.mark.asyncio
    async def test_track_tokens(self, metering_service):
        """Test token tracking"""
        result = await metering_service.track_tokens(
            "board-123", 100, 50, "mixtral-8x7b-32768"
        )
        
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["credits_used"] == 1  # 150 / 100
    
    @pytest.mark.asyncio
    async def test_get_usage_summary(self, metering_service):
        """Test usage summary retrieval"""
        summary = await metering_service.get_usage_summary("board-123")
        
        assert isinstance(summary, UsageSummary)
        assert summary.actionboard_id == "board-123"
        assert summary.credit_limit == 1000
        assert summary.token_limit == 100000
    
    def test_operation_credits_mapping(self, metering_service):
        """Test operation credits are correct"""
        assert metering_service.OPERATION_CREDITS["email_analysis"] == 5
        assert metering_service.OPERATION_CREDITS["document_processing"] == 8
        assert metering_service.OPERATION_CREDITS["data_transformation"] == 3
        assert metering_service.OPERATION_CREDITS["api_integration"] == 6
        assert metering_service.OPERATION_CREDITS["image_processing"] == 10


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestGroqColabAgent:
    """Test main agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes"""
        assert agent is not None
        assert agent.model == "mixtral-8x7b-32768"
        assert agent.total_requests == 0
        assert agent.successful_requests == 0
    
    def test_validate_task(self, agent, sample_task):
        """Test task validation"""
        assert agent._validate_task(sample_task) is True
    
    def test_validate_task_fails(self, agent):
        """Test task validation fails"""
        invalid_task = {"actionboard_id": "board-123"}  # Missing prompt
        assert agent._validate_task(invalid_task) is False
    
    def test_get_performance_summary(self, agent):
        """Test performance summary"""
        summary = agent.get_performance_summary()
        
        assert "total_requests" in summary
        assert "successful_requests" in summary
        assert "failed_requests" in summary
        assert "success_rate_pct" in summary
        assert "avg_latency_ms" in summary
    
    @pytest.mark.asyncio
    async def test_process_task_invalid_api_key(self, sample_task):
        """Test task processing with invalid API key"""
        agent = GroqColabAgent(api_key="invalid-key")
        # This will fail due to API error
        result = await agent.process_task(sample_task)
        
        # Should handle error gracefully
        assert "status" in result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATA MODEL TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDataModels:
    """Test data models"""
    
    def test_metering_data_creation(self, sample_metering_data):
        """Test MeteringData creation"""
        assert sample_metering_data.action_id == "action-001"
        assert sample_metering_data.credits_used == 5
        assert sample_metering_data.total_tokens == 150
    
    def test_metering_data_conversion(self, sample_metering_data):
        """Test MeteringData conversion to dict"""
        from dataclasses import asdict
        data_dict = asdict(sample_metering_data)
        
        assert isinstance(data_dict, dict)
        assert data_dict["action_id"] == "action-001"
    
    def test_usage_summary_creation(self):
        """Test UsageSummary creation"""
        summary = UsageSummary(
            actionboard_id="board-123",
            current_credits=500,
            credit_limit=1000,
            credits_percentage=50.0,
            current_tokens=50000,
            token_limit=100000,
            tokens_percentage=50.0,
            reset_date=datetime.now() + timedelta(days=30),
            daily_avg_usage=100.0,
            projected_end_date=datetime.now() + timedelta(days=30)
        )
        
        assert summary.actionboard_id == "board-123"
        assert summary.credits_percentage == 50.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALIZATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestVisualization:
    """Test visualization components"""
    
    def test_print_header(self, capsys):
        """Test header printing"""
        VisualAnalytics.print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
    
    def test_print_comparison_chart(self, capsys):
        """Test comparison chart"""
        VisualAnalytics.print_comparison_chart()
        captured = capsys.readouterr()
        assert "Groq Mixtral" in captured.out
        assert "OpenAI" in captured.out
    
    def test_print_error_rates(self, capsys):
        """Test error rate visualization"""
        VisualAnalytics.print_error_rates()
        captured = capsys.readouterr()
        assert "Success Rate" in captured.out
    
    def test_print_key_metrics(self, capsys):
        """Test key metrics display"""
        VisualAnalytics.print_key_metrics()
        captured = capsys.readouterr()
        assert "45ms" in captured.out


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: DASHBOARD TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDashboard:
    """Test dashboard"""
    
    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, agent):
        """Test dashboard initializes"""
        dashboard = Dashboard(agent)
        assert dashboard is not None
        assert dashboard.agent == agent
    
    @pytest.mark.asyncio
    async def test_display_main_dashboard(self, agent, capsys):
        """Test main dashboard display"""
        dashboard = Dashboard(agent)
        dashboard.display_main_dashboard()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, storage, config):
        """Test complete workflow"""
        # Initialize
        agent = GroqColabAgent(
            api_key="test-key",
            model="mixtral-8x7b-32768",
            config=config
        )
        
        # Validate task
        task = {
            "actionboard_id": "board-123",
            "prompt": "Test",
            "action_type": "monitoring"
        }
        assert agent._validate_task(task) is True
        
        # Get performance
        summary = agent.get_performance_summary()
        assert summary["total_requests"] == 0
    
    def test_storage_workflow(self, storage, sample_metering_data):
        """Test storage workflow"""
        # Insert metering
        result = storage.insert_metering(sample_metering_data)
        assert result is True
        
        # Retrieve data
        billing_data = storage.get_billing_data("board-123")
        assert isinstance(billing_data, list)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: PERFORMANCE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance tests"""
    
    def test_config_load_time(self, benchmark):
        """Benchmark config loading"""
        result = benchmark(ConfigManager)
        assert result is not None
    
    def test_storage_insert_time(self, benchmark, storage, sample_metering_data):
        """Benchmark storage insert"""
        result = benchmark(
            storage.insert_metering,
            sample_metering_data
        )
        assert result is True


# ═════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

pytest_plugins = ['pytest_asyncio']

# Marks
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# Run tests with: pytest tests/ -v --cov=. --cov-report=html
# Run specific test: pytest tests/test_agent.py::TestConfigManager::test_config_initialization -v
# Run with markers: pytest -m "not performance" -v
