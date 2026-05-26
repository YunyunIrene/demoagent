import asyncio
import logging
import time
from enum import Enum
from typing import Optional
from copilot import CopilotClient, SubprocessConfig
from copilot.session import PermissionHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Harness")

#定义状态机
class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

#  Harness 类
class Harness:
    #为 Agent 添加日志、状态、失败记录的控制层
    def __init__(self, agent):
        self.agent = agent
        self.state = AgentState.IDLE
        self.last_error: Optional[str] = None
        self.call_count = 0
        self.failure_count = 0

    async def run(self, task: str) -> str:
        #带 harness 的任务执行入口
        self.call_count += 1
        call_id = self.call_count
        logger.info(f"[Call #{call_id}] Starting task: {task[:50]}...")
        
        # 状态转换: IDLE → RUNNING
        self._set_state(AgentState.RUNNING)
        start_time = time.perf_counter()
        
        try:
            # 调用真正的 agent 逻辑
            result = await self.agent._process_task_impl(task)
            
            # 成功: 记录耗时，更新状态
            elapsed = time.perf_counter() - start_time
            logger.info(f"[Call #{call_id}] Succeeded in {elapsed:.3f}s")
            self._set_state(AgentState.SUCCEEDED)
            return result
            
        except Exception as e:
            # 失败: 记录错误，更新状态
            elapsed = time.perf_counter() - start_time
            self.last_error = str(e)
            self.failure_count += 1
            logger.error(f"[Call #{call_id}] Failed after {elapsed:.3f}s: {self.last_error}")
            self._set_state(AgentState.FAILED)
            # 可以选择重新抛出，或者返回一个降级结果
            raise   # 保持原有异常传播

    def _set_state(self, new_state: AgentState):
        #状态变更 + 日志
        old_state = self.state
        self.state = new_state
        logger.debug(f"State transition: {old_state.value} -> {new_state.value}")

    def get_stats(self) -> dict:
        #获取 harness 统计信息
        return {
            "state": self.state.value,
            "total_calls": self.call_count,
            "failures": self.failure_count,
            "last_error": self.last_error,
        }

#原始 Agent 业务逻辑（保持纯净）
class SimpleAgent:
    def __init__(self):
        self.client = None
        self.session = None
        self.conversation_history = []
        # 注意：我们把 harness 作为包装器，不污染 agent 本身
        self.harness = Harness(self)   # 互相引用，但清晰

    async def initialize(self):
        #添加日志
        logger.info("Initializing Copilot client...")
        #初始化部分保持不变
        config = SubprocessConfig(github_token="Your token")
        self.client = CopilotClient(config)
        await self.client.start()
        
        self.session = await self.client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model="gpt-4.1"
        )
        #添加日志
        logger.info("Agent initialized successfully.")

    # 核心业务逻辑，与之前的process_task完全相同
    async def _process_task_impl(self, task: str) -> str:
        # 创建包含历史记录的消息上下文
        # 只取最近的100条消息，防止超出上下文长度限制
        recent_history = self.conversation_history[-100:]
        
        # 构建完整的提示，包含历史记录
        context_messages = []
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            context_messages.append(f"{role.capitalize()}: {content}")
        
        # 添加当前任务
        context_messages.append(f"User: {task}")
        context_messages.append("Assistant:")  # 提示AI继续回应
        
        # 将上下文组合成一个字符串
        full_context = "\n".join(context_messages)
        
        # 发送带有上下文的请求
        response = await self.session.send_and_wait(prompt=full_context)
        response_content = response.data.content
        
        # 更新对话历史
        self.conversation_history.append({"role": "user", "content": task})
        self.conversation_history.append({"role": "assistant", "content": response_content})
        
        return response_content
        

    # 对外暴露的接口，通过 harness 调用
    async def process_task(self, task: str) -> str:
        return await self.harness.run(task)

    async def run_conversation(self):
        #交互循环，加入try-except机制
        print("\n🤖 Agent with Harness is ready. Type 'quit' to exit.\n")
        while True:
            user_input = input("请输入问题 (输入 'quit' 退出): ")
            if user_input.lower() == 'quit':
                break
            try:
                response = await self.process_task(user_input)
                print(f"AI: {response}\n")
            except Exception:
                print(f"⚠️  Task failed. Harness stats: {self.harness.get_stats()}\n")

    async def cleanup(self):
        #cleanup逻辑不变
        if self.client:
            await self.client.stop()
            #增加log
            logger.info("Agent cleaned up.")

# 使用示例
async def main():
    agent = SimpleAgent()
    await agent.initialize()
    await agent.run_conversation()
    await agent.cleanup()
    
    # 最终打印 harness 统计
    print("\n📊 Final Harness Statistics:")
    for k, v in agent.harness.get_stats().items():
        print(f"   {k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
