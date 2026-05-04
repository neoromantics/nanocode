import os
import re
import fnmatch
import asyncio
import tempfile
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

class ToolResult:
    def __init__(self, value: Optional[str] = None, error: Optional[str] = None):
        self.value = value
        self.error = error

    def is_ok(self) -> bool:
        return self.error is None

class Tool(ABC):
    @abstractmethod
    def name(self) -> str: pass
    
    @abstractmethod
    def description(self) -> str: pass
    
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]: pass
    
    @abstractmethod
    async def execute(self, args: Dict[str, Any]) -> ToolResult: pass

class ReadTool(Tool):
    def name(self) -> str: return "read"
    def description(self) -> str: return "Read file with line numbers (file path, not directory)"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"}
            },
            "required": ["path"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        offset = args.get("offset", 0)
        limit = args.get("limit", -1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if limit < 0:
                limit = len(lines) - offset
            
            if offset < 0: offset = 0
            if offset >= len(lines): return ToolResult(value="")
            
            sliced = lines[offset : offset + limit]
            result = ""
            for i, line in enumerate(sliced, start=offset + 1):
                result += f"{i:4}| {line}"
            if not result.endswith("\n") and result: result += "\n"
            return ToolResult(value=result)
        except Exception as e:
            return ToolResult(error=f"error: could not open {path}: {str(e)}")

class WriteTool(Tool):
    def name(self) -> str: return "write"
    def description(self) -> str: return "Write content to file"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(value="ok")
        except Exception as e:
            return ToolResult(error=f"error: could not write {path}: {str(e)}")

class EditTool(Tool):
    def name(self) -> str: return "edit"
    def description(self) -> str: return "Replace old with new in file (old must be unique unless all=true)"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "all": {"type": "boolean"}
            },
            "required": ["path", "old", "new"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        path = str(args.get("path", ""))
        old_str = str(args.get("old", ""))
        new_str = str(args.get("new", ""))
        replace_all = bool(args.get("all", False))
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            
            count = text.count(old_str)
            if count == 0:
                return ToolResult(error="error: old_string not found")
            if count > 1 and not replace_all:
                return ToolResult(error=f"error: old_string appears {count} times, must be unique (use all=true)")
            
            if replace_all:
                text = text.replace(old_str, new_str)
            else:
                text = text.replace(old_str, new_str, 1)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return ToolResult(value="ok")
        except Exception as e:
            return ToolResult(error=f"error: could not edit {path}: {str(e)}")

class GlobTool(Tool):
    def name(self) -> str: return "glob"
    def description(self) -> str: return "Find files by pattern, sorted by mtime"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pat": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pat"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        pat = args.get("pat")
        start_path = args.get("path", ".")
        if not start_path: start_path = "."
        
        try:
            results = []
            for root, dirs, files in os.walk(start_path):
                for name in files:
                    rel_path = os.path.relpath(os.path.join(root, name), start_path)
                    if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(name, pat):
                        results.append(os.path.join(root, name))
            
            if not results: return ToolResult(value="none")
            
            # Sort by mtime descending
            results.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return ToolResult(value="\n".join(results))
        except Exception as e:
            return ToolResult(error=f"error: {str(e)}")

class GrepTool(Tool):
    def name(self) -> str: return "grep"
    def description(self) -> str: return "Search files for regex pattern"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pat": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pat"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        pat = args.get("pat")
        start_path = args.get("path", ".")
        try:
            regex = re.compile(pat)
            hits = []
            for root, dirs, files in os.walk(start_path):
                for name in files:
                    full_path = os.path.join(root, name)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    hits.append(f"{full_path}:{i}:{line.strip()}")
                                    if len(hits) >= 50: break
                    except (IOError, OSError, UnicodeDecodeError):
                        continue
                    if len(hits) >= 50: break
                if len(hits) >= 50: break
            
            if not hits: return ToolResult(value="none")
            return ToolResult(value="\n".join(hits))
        except Exception as e:
            return ToolResult(error=f"error: {str(e)}")

class BashTool(Tool):
    def name(self) -> str: return "bash"
    def description(self) -> str: return "Run shell command"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        cmd = args.get("cmd", "")
        return await SubprocessHelper.run(cmd)

class FetchUrlTool(Tool):
    def name(self) -> str: return "fetch_url"
    def description(self) -> str: return "Fetch text content from a URL"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        url = args.get("url")
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                text = resp.text
                if len(text) > 100000:
                    text = text[:100000] + "\n...[TRUNCATED]"
                return ToolResult(value=text if text else "(empty or error)")
        except Exception as e:
            return ToolResult(error=f"error: {str(e)}")

class PythonTool(Tool):
    def name(self) -> str: return "execute_python"
    def description(self) -> str: return "Execute a python script and return its stdout/stderr"
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"]
        }
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        code = args.get("code", "")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        
        try:
            res = await SubprocessHelper.run(f"python3 {tmp_path}", prefix="py: ")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return res
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return ToolResult(error=f"error: {str(e)}")

class ToolRegistry:
    _instance = None
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_defaults()

    @classmethod
    def instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_defaults(self):
        self.register_tool(ReadTool())
        self.register_tool(WriteTool())
        self.register_tool(EditTool())
        self.register_tool(GlobTool())
        self.register_tool(GrepTool())
        self.register_tool(BashTool())
        self.register_tool(FetchUrlTool())
        self.register_tool(PythonTool())

    def register_tool(self, tool: Tool):
        self.tools[tool.name()] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name(),
                "description": t.description(),
                "input_schema": t.input_schema()
            }
            for t in self.tools.values()
        ]

class SubprocessHelper:
    @staticmethod
    async def run(cmd: str, prefix: str = "") -> ToolResult:
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            output = []
            while True:
                if not process.stdout: break
                line = await process.stdout.readline()
                if not line: break
                line_str = line.decode(errors="replace")
                print(f"  \033[2m│ {prefix}{line_str}\033[0m", end="", flush=True)
                output.append(line_str)
            
            await process.wait()
            result = "".join(output).strip()
            return ToolResult(value=result if result else "(empty)")
        except Exception as e:
            return ToolResult(error=f"error: {str(e)}")
