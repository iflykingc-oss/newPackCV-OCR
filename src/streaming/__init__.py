"""SSE 流式响应管理器"""
import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Optional


class EventStream:
    """单个事件的 SSE 流"""

    def __init__(self, event_id: str):
        self.event_id = event_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, event_type: str, data: Any) -> None:
        if self.closed:
            return
        await self.queue.put({
            "id": f"{self.event_id}-{int(time.time()*1000)}",
            "event": event_type,
            "data": data,
            "ts": time.time(),
        })

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item


class SSEManager:
    """SSE 事件管理器"""

    def __init__(self):
        self._streams: Dict[str, EventStream] = {}
        self._by_tag: Dict[str, set] = defaultdict(set)  # tag -> stream_ids
        self._stats: Dict[str, int] = defaultdict(int)

    def create_stream(self, tag: Optional[str] = None) -> EventStream:
        """创建事件流"""
        sid = f"sse-{uuid.uuid4().hex[:12]}"
        stream = EventStream(sid)
        self._streams[sid] = stream
        if tag:
            self._by_tag[tag].add(sid)
        self._stats["streams_created"] += 1
        return stream

    async def broadcast(self, tag: str, event_type: str, data: Any) -> int:
        """向 tag 广播事件"""
        count = 0
        for sid in list(self._by_tag.get(tag, set())):
            stream = self._streams.get(sid)
            if stream and not stream.closed:
                await stream.send(event_type, data)
                count += 1
        self._stats["events_sent"] += 1
        self._stats[f"events_sent:{tag}"] += 1
        return count

    def close_stream(self, stream_id: str) -> bool:
        """关闭流"""
        stream = self._streams.pop(stream_id, None)
        if not stream:
            return False
        for tag_streams in self._by_tag.values():
            tag_streams.discard(stream_id)
        asyncio.create_task(stream.close()) if stream else None
        self._stats["streams_closed"] += 1
        return True

    def get_stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            "active_streams": len(self._streams),
            "stats": dict(self._stats),
            "tags": {tag: len(sids) for tag, sids in self._by_tag.items()},
        }


# 单例
_singleton: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    global _singleton
    if _singleton is None:
        _singleton = SSEManager()
    return _singleton
