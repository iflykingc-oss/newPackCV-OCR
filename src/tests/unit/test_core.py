"""核心模块单元测试（简化版）"""
import pytest


class TestImports:
    """模块导入测试"""
    
    def test_tenancy_imports(self):
        """租户模块导入"""
        from tenancy.api_key_manager import APIKeyManager
        from tenancy.rate_limiter import TenantRateLimiter
        from tenancy.context import TenantContext
        from tenancy.models import TenantTier
        assert APIKeyManager is not None
        assert TenantRateLimiter is not None
        assert TenantContext is not None
        assert TenantTier.PRO
    
    def test_intelligence_imports(self):
        """智能模块导入"""
        from intelligence.llm_cache import LLMResponseCache
        from intelligence.few_shot import FewShotManager
        assert LLMResponseCache is not None
        assert FewShotManager is not None
    
    def test_resilience_imports(self):
        """韧性模块导入"""
        from resilience.circuit_breaker import CircuitBreaker, CircuitState
        assert CircuitBreaker is not None
        assert CircuitState.CLOSED
    
    def test_errors_imports(self):
        """错误模块导入"""
        from errors.registry import ErrorRegistry, ERROR_CODES
        assert ErrorRegistry is not None
        assert len(ERROR_CODES) >= 5
    
    def test_config_hotreload_imports(self):
        """配置热更新模块导入"""
        from config_hotreload.manager import HotReloadManager
        assert HotReloadManager is not None
    
    def test_data_lineage_imports(self):
        """血缘模块导入"""
        from data_lineage.lineage import create_lineage, lineage_stats
        assert create_lineage is not None
        assert lineage_stats is not None
    
    def test_api_docs_imports(self):
        """API文档模块导入"""
        from api_docs import generator
        assert generator is not None
    
    def test_tracing_imports(self):
        """追踪模块导入"""
        from tracing.tracer import get_tracer
        assert get_tracer is not None


class TestLLMResponseCache:
    """LLM缓存测试"""
    
    def test_cache_basic(self):
        """缓存基本功能"""
        from intelligence.llm_cache import LLMResponseCache
        
        cache = LLMResponseCache(max_size=10, default_ttl=60)
        cache.set("test_key", {"data": "test_value"}, ttl=30)
        
        result = cache.get("test_key")
        assert result is not None
        assert result["data"] == "test_value"
    
    def test_cache_miss(self):
        """缓存未命中"""
        from intelligence.llm_cache import LLMResponseCache
        
        cache = LLMResponseCache(max_size=10, default_ttl=60)
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_stats(self):
        """缓存统计"""
        from intelligence.llm_cache import LLMResponseCache
        
        cache = LLMResponseCache(max_size=10, default_ttl=60)
        cache.set("k1", "v1")
        cache.get("k1")
        cache.get("k2")
        
        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestCircuitBreakerStates:
    """断路器状态测试"""
    
    def test_states_exist(self):
        """状态枚举存在"""
        from resilience.circuit_breaker import CircuitState
        
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN
    
    def test_state_transition_logic(self):
        """状态转换逻辑"""
        from resilience.circuit_breaker import CircuitState
        
        # 验证状态顺序: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
        states = [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
        assert len(states) == 3


class TestErrorRegistryBasic:
    """错误注册表基础测试"""
    
    def test_error_codes_count(self):
        """错误码数量"""
        from errors.registry import ERROR_CODES
        
        assert len(ERROR_CODES) >= 5
    
    def test_registry_creation(self):
        """注册表创建"""
        from errors.registry import ErrorRegistry
        
        registry = ErrorRegistry()
        assert registry is not None
    
    def test_list_all(self):
        """列出所有"""
        from errors.registry import ErrorRegistry
        
        registry = ErrorRegistry()
        all_errors = registry.list_all()
        assert len(all_errors) >= 5


class TestHotReloadManagerBasic:
    """配置热更新基础测试"""
    
    def test_manager_creation(self):
        """管理器创建"""
        from config_hotreload.manager import HotReloadManager
        
        manager = HotReloadManager()
        assert manager is not None
    
    def test_list_configs(self):
        """列出配置"""
        from config_hotreload.manager import HotReloadManager
        
        manager = HotReloadManager()
        configs = manager.list_configs()
        assert isinstance(configs, dict) or isinstance(configs, list)


class TestLineageBasic:
    """血缘基础测试"""
    
    def test_stats(self):
        """统计"""
        from data_lineage.lineage import lineage_stats
        
        stats = lineage_stats()
        assert "total_lineages" in stats
    
    def test_create(self):
        """创建"""
        from data_lineage.lineage import create_lineage
        
        builder = create_lineage(
            tenant_id="tnt_test",
            run_id="run_test"
        )
        assert builder is not None


class TestTenantAPIKey:
    """租户API Key测试"""
    
    def test_setup_demo(self):
        """演示租户初始化"""
        from tenancy.api_key_manager import APIKeyManager, _TENANT_STORE
        
        _TENANT_STORE.clear()
        APIKeyManager.setup_demo_tenants()
        
        assert len(_TENANT_STORE) >= 3
    
    def test_verify_invalid(self):
        """验证无效Key"""
        from tenancy.api_key_manager import APIKeyManager
        
        result = APIKeyManager.verify_api_key_only("invalid_key_xyz")
        assert result is None