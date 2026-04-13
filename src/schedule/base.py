"""
Schedule base class - abstract interface for schedulers

所有调度器（EJF、MUSS V2-V7）应继承此类
"""

from abc import ABC, abstractmethod


class ScheduleStrategy(ABC):
    """调度器基类接口"""

    @abstractmethod
    def schedule(self) -> None:
        """执行调度，生成事件序列"""
        pass

    @abstractmethod
    def get_schedule(self):
        """返回调度结果（Schedule 对象）"""
        pass
