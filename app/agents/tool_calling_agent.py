from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_db import DbLogger
from app.mcp.tool_registry import ToolRegistry
from app.agents.system_prompts import system_for_agent

from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat_client import OpenAICompatClient


class ToolCallingAgent:
    def __init__(
            self,
            registry: ToolRegistry,
            db: Optional[DbLogger],
            *,
            llm_provider: str,
            ollama: Optional[OllamaClient] = None,
            openai_compat: Optional[OpenAICompatClient] = None,
            max_steps: int = 2,
    ):
        self.registry = registry
        self.db = db
        self.llm_provider = llm_provider
        self.ollama = ollama
        self.openai_compat = openai_compat
        self.max_steps = max_steps

    def _log(self, event_type: str, payload: Dict[str, Any], *, run_id: str, agent_name: str) -> None:
        try:
            if self.db is not None:
                self.db.log_event(event_type, payload, run_id=run_id, agent_name=agent_name)
        except Exception:
            pass

    def _llm(self, system: str, prompt: str) -> str:
        if self.llm_provider == "ollama":
            if not self.ollama:
                raise RuntimeError("Ollama client not configured")
            return self.ollama.generate(prompt=prompt, system=system)

        if self.llm_provider == "openai_compat":
            if not self.openai_compat:
                raise RuntimeError("OpenAI-compat client not configured")
            return self.openai_compat.chat(system=system, user=prompt)

        raise ValueError(f"Unknown LLM_PROVIDER: {self.llm_provider}")

    @staticmethod
    def _build_prompt(
            *,
            include_tools: bool,
            tools_desc: str,
            message: str,
            results_block: Optional[List[Dict[str, Any]]] = None,
            step: int = 0,
            extra: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if include_tools:
            parts.append(f"TOOLS:\n{tools_desc}\n")
        parts.append(f"USER MESSAGE:\n{message}\n")
        if results_block is not None:
            parts.append(f"TOOL RESULTS (step {step}):\n{json.dumps(results_block, ensure_ascii=False)}\n")
        if extra:
            parts.append(extra.strip() + "\n")
        return "\n".join(parts)

    @staticmethod
    def _looks_like_retrieval_claim(s: str) -> bool:
        s = (s or "").lower()
        patterns = [
            r"\bdoc_id\b",
            r"\bchunk_id\b",
            r"\bsource_type\b",
            r"\bmetadata\b",
            r"\bscore\b",
            r"\bhits\b",
            r"\bretriev",
            r"\bqdrant\b",
            r"\bvector\b",
        ]
        return any(re.search(p, s) for p in patterns)

    @staticmethod
    def _contains_retrieval_artifacts(obj: Any) -> bool:
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            banned = {"doc_id", "chunk_id", "score", "metadata", "source_type", "hits"}
            if keys & banned:
                return True
            return any(ToolCallingAgent._contains_retrieval_artifacts(v) for v in obj.values())

        if isinstance(obj, list):
            return any(ToolCallingAgent._contains_retrieval_artifacts(x) for x in obj)

        if isinstance(obj, str):
            return ToolCallingAgent._looks_like_retrieval_claim(obj)

        return False

    @staticmethod
    def _no_retrieval_response() -> str:
        return "NO_RETRIEVAL"

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None

  
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            text = text[start : end + 1]

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_final_output(out_raw: Any) -> str:
        if out_raw is None:
            return ""

        if isinstance(out_raw, dict):
            if "ts" in out_raw:
                return str(out_raw["ts"])
            return json.dumps(out_raw, ensure_ascii=False)

        if isinstance(out_raw, list):
            return json.dumps(out_raw, ensure_ascii=False)


        s = str(out_raw).strip()
        if not s or s.lower() == "none":
            return ""

        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "ts" in obj:
                    return str(obj["ts"])
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass

        return s

    @staticmethod
    def _message_allows_hits_only(message: str) -> bool:
        m = (message or "").lower()
        return m.startswith("__rag_test__") or "retrieved hits only" in m or "return retrieved hits" in m

    @staticmethod
    def _message_demands_no_retrieval(message: str) -> bool:
        m = (message or "").lower()
        return "output exactly no_retrieval" in m or "return exactly no_retrieval" in m

    def run(
            self,
            message: str,
            *,
            run_id: Optional[str],
            agent_name: str,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        run_id = run_id or str(uuid.uuid4())
        tool_calls: List[Dict[str, Any]] = []

        tools_desc = json.dumps(self.registry.list(), ensure_ascii=False)

        agent_key = (agent_name or "default").lower().strip()
        TOOLS_ENABLED_AGENT = {"default", "fastfinance", "sme_business"}
        tools_allowed = agent_key in TOOLS_ENABLED_AGENT

        msg = (message or "").strip()

        # fast path ping
        if msg.lower() == "ping":
            result = self.registry.call("ping", {})
            tool_calls.append({"name": "ping", "args": {}, "result": result})
            out = self._normalize_final_output(result) or "true"
            return run_id, out, tool_calls

        # fast path rag test
        if msg.startswith("__RAG_TEST__"):
            if not tools_allowed:
                return run_id, self._no_retrieval_response(), tool_calls

            args = {"query": msg, "top_k": 10}
            result = self.registry.call("rag_search", args)
            tool_calls.append({"name": "rag_search", "args": args, "result": result})
            return run_id, json.dumps(result, ensure_ascii=False), tool_calls

        prompt = self._build_prompt(include_tools=tools_allowed, tools_desc=tools_desc, message=message)
        last_results_block: Optional[List[Dict[str, Any]]] = None

        for step in range(self.max_steps):
            system = system_for_agent(agent_name)
            raw = self._llm(system, prompt)
            parsed = self._safe_parse_json(raw)

            if not parsed:
                return run_id, (raw or "").strip() or "I couldn't complete the request.", tool_calls

            msg_type = parsed.get("type")

            if msg_type == "final":
                out_raw = parsed.get("output", "")
                out = self._normalize_final_output(out_raw)

                if not out or out.strip() == "[]" or out.strip().lower() == "none":
                    # never allow blank/None/[] as final
                    out = self._no_retrieval_response() if self._message_demands_no_retrieval(message) else self._no_retrieval_response()

                if not self._message_allows_hits_only(message):
                    if self._contains_retrieval_artifacts(out_raw) or self._contains_retrieval_artifacts(out) or out.strip() == "[]":
                        out = self._no_retrieval_response()
                return run_id, out, tool_calls

            if msg_type == "tool_call":
                if not tools_allowed:
                    return run_id, "I cannot fulfill this request.", tool_calls

                if step > 0:
                    out = self._no_retrieval_response() if self._message_demands_no_retrieval(message) else "I couldn't complete the request."
                    return run_id, out, tool_calls

                calls = parsed.get("calls") or []
                if not isinstance(calls, list):
                    calls = []
                calls = calls[:1]

                if not calls or not isinstance(calls[0], dict):
                    return run_id, self._no_retrieval_response(), tool_calls

                c = calls[0]
                name = c.get("name")
                args = c.get("args") or {}
                if not isinstance(args, dict):
                    args = {}

                # ---- RAG: force query + brand/country when message implies Fast Finance ----
                if name == "rag_search":
                    args = dict(args)
                    args["query"] = message
                    args.setdefault("top_k", 10)

                    m = (message or "").lower()
                    if "fast finance" in m or agent_key in {"fastfinance", "fast_finance"}:
                        args.setdefault("brand", "Fast Finance")
                        args.setdefault("country", "GLOBAL")

                try:
                    result = self.registry.call(name, args)
                    tool_calls.append({"name": name, "args": args, "result": result})
                    results_block = [{"name": name, "ok": True, "result": result}]
                except Exception as e:
                    tool_calls.append({"name": name, "args": args, "error": str(e)})
                    results_block = [{"name": name, "ok": False, "error": str(e)}]

                last_results_block = results_block

                prompt = self._build_prompt(
                    include_tools=False,
                    tools_desc=tools_desc,
                    message=message,
                    results_block=results_block,
                    step=step,
                    extra=(
                        "IMPORTANT:\n"
                        "Use ONLY the tool results above as grounded context.\n"
                        "Do NOT output tool results or hit lists.\n"
                        "If the answer is not explicitly supported, reply exactly: NO_RETRIEVAL.\n"
                        "Return ONLY JSON: {\"type\":\"final\",\"output\":...}\n"
                    ),
                )
                continue

            # unknown type
            out = self._normalize_final_output(parsed.get("output", "")) or self._no_retrieval_response()
            return run_id, out, tool_calls

        # step limit
        return run_id, self._no_retrieval_response(), tool_calls
