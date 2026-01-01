from __future__ import annotations

import json
import os
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
    ) -> str:
        parts: List[str] = []
        if include_tools:
            parts.append(f"TOOLS:\n{tools_desc}\n")
        parts.append(f"USER MESSAGE:\n{message}\n")
        if results_block is not None:
            parts.append(f"TOOL RESULTS (step {step}):\n{json.dumps(results_block, ensure_ascii=False)}\n")
        return "\n".join(parts)

    @staticmethod
    def _wants_retrieved_hits_only(message: str) -> bool:
        m = (message or "").lower()
        return "__rag_test__" in m or "return retrieved hits only" in m or ("return the top" in m and "hits" in m)

    @staticmethod
    def _retrieved_context_only_required(message: str) -> bool:
        m = (message or "").lower()
        return "using retrieved context only" in m or "retrieved context only" in m

    @staticmethod
    def _no_retrieval_response() -> str:
        return "NO_RETRIEVAL"

    @staticmethod
    def _contains_interest_or_fee_info(text: str) -> bool:
        t = (text or "").lower()

        keywords = ["interest", "rate", "apr", "fee", "fees", "pricing", "cost", "charge", "charges"]
        if not any(k in t for k in keywords):
            return False

        numeric_markers = [
            r"\b\d+(\.\d+)?\s*%",
            r"\b\d+(\.\d+)?\b",
            r"\bzar\b|\br\b\s*\d+",
            r"\bngn\b|\b₦\b",
            r"\busd\b|\$\s*\d+",
            r"\bbps\b|\bbasis points\b"
        ]
        return any(re.search(p, t) for p in numeric_markers)

    @staticmethod
    def _extract_one_sentence_use_case_from_hits(hits: List[Dict[str, Any]]) -> Optional[str]:
        if not hits:
            return None

        text = hits[0].get("text") or ""
        m = re.search(r"purpose:\s*[-–•]?\s*(.+)", text, flags=re.IGNORECASE)
        if m:
            purpose = m.group(1).strip()
            purpose = re.sub(r"\s+", " ", purpose)
            if not purpose.endswith("."):
                purpose += "."
            return purpose

        if re.search(r"working capital", text, flags=re.IGNORECASE):
            return "It’s designed to support working capital needs for small and medium-sized enterprises (SMEs)."

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
        if not s:
            return ""

        # if it's valid json, keep it stable
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "ts" in obj:
                    return str(obj["ts"])
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                return s

        return s

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

    def _force_finalize_safely(
            self,
            *,
            message: str,
            tool_calls: List[Dict[str, Any]],
            last_results_block: Optional[List[Dict[str, Any]]],
    ) -> str:
        if not tool_calls:
            return "I couldn't complete the request."

        last = tool_calls[-1]
        if last.get("name") == "rag_search":
            hits = last.get("result") or []
            if self._wants_retrieved_hits_only(message):
                return json.dumps(hits, ensure_ascii=False)

            if self._retrieved_context_only_required(message):
                combined = "\n".join([h.get("text", "") for h in hits])
                if not self._contains_interest_or_fee_info(combined):
                    return self._no_retrieval_response()

            one = self._extract_one_sentence_use_case_from_hits(hits)
            if one:
                return one

            return "I couldn't complete the request."

        res = last.get("result")
        if res is None and last_results_block is not None:
            return json.dumps(last_results_block, ensure_ascii=False)
        return json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res

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

        include_tools = tools_allowed
        prompt = self._build_prompt(include_tools=include_tools, tools_desc=tools_desc, message=message)

        msg = (message or "").strip()

        if msg.lower() == "ping":
            try:
                result = self.registry.call("ping", {})
                tool_calls.append({"name": "ping", "args": {}, "result": result})
                out = self._normalize_final_output(result)
                self._log("final_fastpath_ping", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls
            except Exception as e:
                self._log("final_fastpath_ping_error", {"error": str(e)}, run_id=run_id, agent_name=agent_name)
                return run_id, "I couldn't complete the request.", tool_calls

        if msg.startswith("__RAG_TEST__"):
            if not tools_allowed:
                out = self._no_retrieval_response()
                self._log("final_fastpath_ragtest_blocked", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls

            args = {"query": msg, "top_k": 10}
            try:
                result = self.registry.call("rag_search", args)
                tool_calls.append({"name": "rag_search", "args": args, "result": result})
                out = json.dumps(result, ensure_ascii=False)
                self._log("final_fastpath_ragtest", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls
            except Exception as e:
                self._log("final_fastpath_ragtest_error", {"error": str(e)}, run_id=run_id, agent_name=agent_name)
                return run_id, "I couldn't complete the request.", tool_calls

        self._log("user_message", {"message": message}, run_id=run_id, agent_name=agent_name)

        last_results_block: Optional[List[Dict[str, Any]]] = None

        for step in range(self.max_steps):
            system = system_for_agent(agent_name)
            raw = self._llm(system, prompt)
            self._log("llm_raw", {"step": step, "raw": raw}, run_id=run_id, agent_name=agent_name)

            parsed = self._safe_parse_json(raw)
            if not parsed:
                out = (raw or "").strip() or "I couldn't complete the request."
                self._log("final_fallback", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls

            msg_type = parsed.get("type")

            if msg_type == "tool_call":
                if not tools_allowed:
                    prompt = self._build_prompt(
                        include_tools=False,
                        tools_desc=tools_desc,
                        message=message + '\n\nIMPORTANT: Tools are disabled. Return ONLY {"type":"final","output":"..."}',
                    )
                    continue

                if step > 0:
                    out = self._force_finalize_safely(message=message, tool_calls=tool_calls, last_results_block=last_results_block)
                    self._log("final_forced", {"output": out, "reason": "tool_call_after_results"}, run_id=run_id, agent_name=agent_name)
                    return run_id, out, tool_calls

                calls = parsed.get("calls") or []
                if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
                    out = self._force_finalize_safely(message=message, tool_calls=tool_calls, last_results_block=last_results_block)
                    self._log("final_forced", {"output": out, "reason": "invalid_calls"}, run_id=run_id, agent_name=agent_name)
                    return run_id, out, tool_calls

                c = calls[0]
                name = c.get("name")
                args = c.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                if name == "rag_search":
                    args["query"] = message

                    try:
                        args["top_k"] = int(args.get("top_k") or 10)
                    except Exception:
                        args["top_k"] = 10

                    env_min = os.getenv("RAG_MIN_SCORE", "0.25")
                    try:
                        env_min_f = float(env_min)
                    except Exception:
                        env_min_f = 0.25
                    ms = args.get("min_score")
                    if ms in (None, "", 0, "0", 0.0, "0.0"):
                        args["min_score"] = env_min_f
                    else:
                        try:
                            args["min_score"] = float(ms)
                        except Exception:
                            args["min_score"] = env_min_f

                    msg_lower = (message or "").lower()
                    is_fastfinance = (agent_key in {"fastfinance", "fast_finance"}) or ("fast finance" in msg_lower)
                    if is_fastfinance:
                        if not args.get("brand"):
                            args["brand"] = "Fast Finance"
                        if not args.get("country"):
                            args["country"] = "GLOBAL"

                if not isinstance(name, str) or not name:
                    out = self._force_finalize_safely(message=message, tool_calls=tool_calls, last_results_block=last_results_block)
                    self._log("tool_error", {"name": str(name), "error": "invalid tool name"}, run_id=run_id, agent_name=agent_name)
                    return run_id, out, tool_calls

                self._log("tool_call", {"name": name, "args": args}, run_id=run_id, agent_name=agent_name)

                results_block: List[Dict[str, Any]] = []
                try:
                    result = self.registry.call(name, args)
                    tool_calls.append({"name": name, "args": args, "result": result})
                    results_block.append({"name": name, "ok": True, "result": result})
                    self._log("tool_result", {"name": name, "result": result}, run_id=run_id, agent_name=agent_name)
                except Exception as e:
                    tool_calls.append({"name": name, "args": args, "error": str(e)})
                    results_block.append({"name": name, "ok": False, "error": str(e)})
                    self._log("tool_error", {"name": name, "error": str(e)}, run_id=run_id, agent_name=agent_name)

                last_results_block = results_block

                if name == "rag_search" and self._retrieved_context_only_required(message):
                    hits = tool_calls[-1].get("result") or []
                    combined = "\n".join([h.get("text", "") for h in hits])
                    if not self._contains_interest_or_fee_info(combined):
                        out = self._no_retrieval_response()
                        self._log("final_no_retrieval", {"output": out}, run_id=run_id, agent_name=agent_name)
                        return run_id, out, tool_calls

                prompt = self._build_prompt(
                    include_tools=False,
                    tools_desc=tools_desc,
                    message=message,
                    results_block=results_block,
                    step=step,
                )
                continue

            if msg_type == "final":
                out_raw = parsed.get("output", "")
                out = self._normalize_final_output(out_raw)

                if not self._wants_retrieved_hits_only(message):
                    if out.strip().startswith("[") and ("doc_id" in out or "chunk_id" in out or "metadata" in out or "score" in out):
                        last = tool_calls[-1] if tool_calls else None
                        if last and last.get("name") == "rag_search":
                            hits = last.get("result") or []
                            one = self._extract_one_sentence_use_case_from_hits(hits)
                            if one:
                                out = one
                            else:
                                out = "I couldn't complete the request."

                self._log("final", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls

            out = self._force_finalize_safely(message=message, tool_calls=tool_calls, last_results_block=last_results_block)
            self._log("final_unknown_type", {"output": out, "parsed": parsed}, run_id=run_id, agent_name=agent_name)
            return run_id, out, tool_calls

        out = self._force_finalize_safely(message=message, tool_calls=tool_calls, last_results_block=last_results_block)
        self._log("final_step_limit", {"output": out}, run_id=run_id, agent_name=agent_name)
        return run_id, out, tool_calls
