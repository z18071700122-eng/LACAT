"""
LLM调用模块：支持API模式和本地模型模式

使用方法:
1. API模式（默认）: LLM(model_name="gpt-3.5-turbo", mode="api")
2. 本地模式: LLM(model_name="qwen3", mode="local", local_model_path="../LLM/qwen3/")
"""
import os
import re
import time
import logging
from typing import Optional, Literal, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_calls.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 检查依赖可用性
# ============================================================

# 本地模式依赖
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LOCAL_MODE_AVAILABLE = True
except ImportError:
    LOCAL_MODE_AVAILABLE = False
    logger.info("transformers/torch未安装，本地模式不可用")

# API模式依赖
try:
    from openai import OpenAI
    API_MODE_AVAILABLE = True
except ImportError:
    API_MODE_AVAILABLE = False
    logger.warning("openai库未安装，API模式不可用")


# ============================================================
# 默认配置
# ============================================================

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MAX_RETRY_ATTEMPTS = 5
RETRY_DELAY = 2

# 全局LLM配置（可通过 set_llm_config 修改）
_llm_global_config: Dict[str, Any] = {
    "mode": "api",
    "local_model_path": "../LLM/qwen3/",
    "local_device": "cuda:0",
}


def set_llm_config(config: Dict[str, Any]):
    """
    设置全局LLM配置
    
    Args:
        config: 配置字典，支持的键:
            - mode: "api" 或 "local"
            - local_model_path: 本地模型路径
            - local_device: 本地模型设备
    """
    global _llm_global_config
    _llm_global_config.update(config)
    logger.info(f"LLM全局配置已更新: {_llm_global_config}")


def get_llm_config() -> Dict[str, Any]:
    """获取当前全局LLM配置"""
    return _llm_global_config.copy()


# ============================================================
# LLM类：支持API和本地双模式
# ============================================================

class LLM:
    """
    语言模型接口类，支持API模式和本地模型模式
    
    API模式：调用OpenAI兼容的API接口
    本地模式：使用transformers加载本地模型（如Qwen3）
    """
    
    # 类变量：缓存本地模型（避免重复加载）
    _local_model_cache = None
    _local_tokenizer_cache = None
    _local_model_path_cache = None
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = MAX_RETRY_ATTEMPTS,
        # 新增参数：支持本地模式
        mode: Optional[Literal["api", "local"]] = None,
        local_model_path: Optional[str] = None,
        local_device: Optional[str] = None,
    ):
        """
        初始化LLM接口
        
        Args:
            model_name: 模型名称
            api_key: API密钥（API模式）
            api_base: API基础URL（API模式）
            max_retries: 最大重试次数
            mode: 运行模式 ("api" 或 "local")，None则使用全局配置
            local_model_path: 本地模型路径
            local_device: 本地模型设备
        """
        self.model_name = model_name
        self.max_retries = max_retries
        
        # 使用传入参数或全局配置
        self.mode = mode or _llm_global_config.get("mode", "api")
        self.local_model_path = local_model_path or _llm_global_config.get("local_model_path", "../LLM/qwen3/")
        self.local_device = local_device or _llm_global_config.get("local_device", "cuda:0")
        
        # API相关
        self.api_key = api_key or DEFAULT_API_KEY
        self.api_base = api_base or DEFAULT_API_BASE
        self.client = None
        
        # 根据模式初始化
        if self.mode == "local":
            if not LOCAL_MODE_AVAILABLE:
                logger.warning("本地模式依赖不可用，切换到API模式")
                self.mode = "api"
            else:
                self._init_local_model()
        
        if self.mode == "api":
            if not API_MODE_AVAILABLE:
                raise RuntimeError("API模式依赖(openai库)未安装")
            self._init_api_client()
        
        logger.info(f"初始化LLM接口，模型: {model_name}，模式: {self.mode}")
    
    def _init_api_client(self):
        """初始化API客户端"""
        if not self.api_key:
            raise ValueError("API密钥未设置。请设置OPENAI_API_KEY环境变量或直接提供api_key参数")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        logger.info(f"API客户端初始化成功，端点: {self.api_base}")
    
    def _init_local_model(self):
        """初始化本地模型（使用缓存避免重复加载）"""
        # 检查是否已缓存相同路径的模型
        if (LLM._local_model_cache is not None and 
            LLM._local_model_path_cache == self.local_model_path):
            logger.info("使用缓存的本地模型")
            return
        
        try:
            logger.info(f"正在加载本地模型: {self.local_model_path}")
            
            LLM._local_tokenizer_cache = AutoTokenizer.from_pretrained(
                self.local_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                device_map = self.local_device
                dtype = torch.float16
            else:
                device_map = "cpu"
                dtype = torch.float32
                logger.warning("GPU不可用，使用CPU（速度较慢）")
            
            LLM._local_model_cache = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=dtype,
            )
            
            LLM._local_model_cache.eval()
            LLM._local_model_path_cache = self.local_model_path
            
            logger.info(f"✓ 本地模型加载成功，设备: {device_map}")
            
        except Exception as e:
            logger.error(f"✗ 本地模型加载失败: {e}")
            logger.info("切换到API模式")
            self.mode = "api"
            self._init_api_client()
    
    def get_output(self, prompt: str, system_prompt: str = "你是一名教育专家") -> str:
        """
        从语言模型获取输出
        
        Args:
            prompt: 提示文本
            system_prompt: 系统提示
            
        Returns:
            模型生成的回复
        """
        if self.mode == "local":
            return self._get_local_output(prompt, system_prompt)
        else:
            return self._get_api_output(prompt, system_prompt)
    
    def _get_local_output(self, prompt: str, system_prompt: str) -> str:
        """调用本地模型获取输出"""
        try:
            start_time = time.time()
            
            model = LLM._local_model_cache
            tokenizer = LLM._local_tokenizer_cache
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            
            # 解码响应
            response_ids = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            duration = time.time() - start_time
            logger.info(f"本地模型调用成功，耗时: {duration:.2f}秒")
            
            return self._clean_response(response)
            
        except Exception as e:
            logger.error(f"本地模型调用失败: {e}")
            return f'{{"error": "本地模型调用失败", "details": "{str(e)}"}}'
    
    def _get_api_output(self, prompt: str, system_prompt: str) -> str:
        """调用API获取输出，包含重试逻辑"""
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                logger.debug(f"调用API，尝试 #{attempt+1}")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
                duration = time.time() - start_time
                logger.info(f"API调用成功，耗时: {duration:.2f}秒")
                
                output = response.choices[0].message.content
                return self._clean_response(output)
                
            except Exception as e:
                attempt += 1
                last_exception = e
                logger.warning(f"API调用失败 (尝试 {attempt}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries:
                    sleep_time = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
        
        logger.error(f"所有API调用尝试均失败: {str(last_exception)}")
        return '{"error": "API调用失败", "details": "' + str(last_exception).replace('"', '\\"') + '"}'
    
    def _clean_response(self, text: str) -> str:
        """清理响应文本"""
        if not text:
            return ""
        
        # 过滤<think>标签（Qwen3特有的思考标签）
        text = re.sub(r'<\s*think\s*>.*?</\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        
        # 去除特殊控制字符
        text = re.sub(r"[\u0010-\u001F\u007F-\u009F]", "", text)
        
        # 去除代码块标记
        text = re.sub(r"```(?:json|python|text)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        
        # 去除多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        
        return text