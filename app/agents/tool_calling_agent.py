from __future__ import annotations

import json
import uuid
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_db import DbLogger
from app.mcp.tool_registry import ToolRegistry
from app.agents.system_prompts import system_for_agent

from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat_client import OpenAICompatClient


class ToolCallingAgent:
    """
    Deterministic 2-step tool loop:

      step 0: model may return {"type":"tool_call", ...} OR {"type":"final", ...}
      step 1: after tool results are provided, model MUST return {"type":"final", ...}

    Hard-enforced:
      - at most ONE tool call
      - no tool calls after results
      - if model misbehaves, we still return something sensible
    """

    def __init__(
            self,
            registry: ToolRegistry,
            db: DbLogger,
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
    def _looks_like_retrieval_claim(s: str) -> bool:
        """

        :param s:
        :return:
        """

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
            r"\bvector\b"
        ]
        return any(re.search(p, s) for p in patterns)


    @staticmethod
    def _no_retrieval_response() -> str:
        return "NO_RETRIEVAL"


    @staticmethod
    def _contains_retrieval_artifacts(obj: Any) -> bool:
        """
        True if obj (dict/list) contains keys that look like retrieval evidence.
        Works even when the model returns a dict (not just a string).
        """
        if isinstance(obj, dict):
            # direct keys
            keys = {str(k).lower() for k in obj.keys()}
            banned = {"doc_id", "chunk_id", "score", "metadata", "source_type", "hits"}
            if keys & banned:
                return True
            # nested
            return any(ToolCallingAgent._contains_retrieval_artifacts(v) for v in obj.values())

        if isinstance(obj, list):
            return any(ToolCallingAgent._contains_retrieval_artifacts(x) for x in obj)

        if isinstance(obj, str):
            # fallback for strings
            return ToolCallingAgent._looks_like_retrieval_claim(obj)

        return False

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
        TOOLS_ENABLED_AGENT = {
            "default",
            "fastfinance",
            "sme_business"
        }
        tools_allowed = agent_key in TOOLS_ENABLED_AGENT  # only default agent can use tools
        include_tools = tools_allowed
        prompt = self._build_prompt(include_tools=include_tools, tools_desc=tools_desc, message=message)


        self.db.log_event("user_message", {"message": message}, run_id=run_id, agent_name=agent_name)

        last_results_block: Optional[List[Dict[str, Any]]] = None

        for step in range(self.max_steps):
            from app.agents.system_prompts import system_for_agent
            system = system_for_agent(agent_name)
            raw = self._llm(system, prompt)
            self.db.log_event("llm_raw", {"step": step, "raw": raw}, run_id=run_id, agent_name=agent_name)

            parsed = self._safe_parse_json(raw)
            if not parsed:
                out = raw.strip()
                self.db.log_event("final_fallback", {"output": out}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls

            msg_type = parsed.get("type")

            if msg_type not in {"final", "tool_call"}:
                out_raw = parsed.get("output", parsed)
                out = self._normalize_final_output(out_raw, tool_calls)
                if not tool_calls and (
                    self._contains_retrieval_artifacts(out_raw) or self._contains_retrieval_artifacts(out)
                ):
                    blocked = self._no_retrieval_response()
                    self.db.log_event(
                        "final_blocked_fake_sources",
                        {"raw": out_raw, "normalized": out, "output": blocked},
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                    return run_id, blocked, tool_calls

                self.db.log_event("final_unknown_type_coerced", {"output": out, "parsed" : parsed}, run_id=run_id, agent_name=agent_name)
                return run_id, out, tool_calls

                # If this agent forbids tools and the model tries to call one,
            # re-ask for a final answer (no tools).
            if msg_type == "tool_call" and not tools_allowed:
                prompt2 = self._build_prompt(
                    include_tools=False,
                    tools_desc=tools_desc,
                    message=message + "\n\nIMPORTANT: Tools are disabled for this agent. "
                                      "Return ONLY: {\"type\":\"final\",\"output\":...}",
                )
                raw2 = self._llm(system, prompt2)
                self.db.log_event("llm_raw", {"step": step, "raw": raw2, "retry": True},
                                  run_id=run_id, agent_name=agent_name)

                parsed2 = self._safe_parse_json(raw2)
                if parsed2 and parsed2.get("type") == "final":
                    out2 = self._normalize_final_output(parsed2.get("output", ""), tool_calls)
                    self.db.log_event("final", {"output": out2}, run_id=run_id, agent_name=agent_name)
                    return run_id, out2, tool_calls

                # Fallback: return raw model text
                out2 = (raw2 or "").strip() or "I couldn't complete the request."
                self.db.log_event("final_fallback", {"output": out2}, run_id=run_id, agent_name=agent_name)
                return run_id, out2, tool_calls


            if msg_type == "tool_call":
                # no tool calls allowed after results
                if step > 0:
                    out = self._force_finalize_from_last_tool(tool_calls, last_results_block)
                    self.db.log_event(
                        "final_forced",
                        {"output": out, "reason": "tool_call_after_results"},
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                    return run_id, out, tool_calls

                calls = parsed.get("calls") or []
                if not isinstance(calls, list):
                    calls = []

                # at most ONE tool call
                calls = calls[:1]

                results_block: List[Dict[str, Any]] = []

                if not calls:
                    out = self._force_finalize_from_last_tool(tool_calls, results_block)
                    self.db.log_event(
                        "final_forced",
                        {"output": out, "reason": "empty_tool_calls"},
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                    return run_id, out, tool_calls

                c = calls[0]
                if not isinstance(c, dict):
                    out = self._force_finalize_from_last_tool(tool_calls, results_block)
                    self.db.log_event(
                        "final_forced",
                        {"output": out, "reason": "non_dict_tool_call"},
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                    return run_id, out, tool_calls

                name = c.get("name")
                args = c.get("args") or {}

                if name == "rag_search":
                    args = dict(args or {})
                    args["query"] = message

                    if "top_k" not in args or not args["top_k"]:
                        args["top_k"] = 10

                    agent_key = (agent_name or "default").lower().strip()
                    if agent_key in {"fastfinance", "fast_finance"}:
                        args.setdefault("brand", "Fast Finance")
                        args.setdefault("country", "GLOBAL")


                if not isinstance(name, str) or not name:
                    self.db.log_event(
                        "tool_error",
                        {"name": str(name), "error": "invalid tool name"},
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                    out = self._force_finalize_from_last_tool(tool_calls, results_block)
                    return run_id, out, tool_calls

                if not isinstance(args, dict):
                    args = {}

                self.db.log_event("tool_call", {"name": name, "args": args}, run_id=run_id, agent_name=agent_name)

                try:
                    result = self.registry.call(name, args)
                    self.db.log_event(
                        "tool_result", {"name": name, "result": result}, run_id=run_id, agent_name=agent_name
                    )
                    tool_calls.append({"name": name, "args": args, "result": result})
                    results_block.append({"name": name, "ok": True, "result": result})
                except Exception as e:
                    self.db.log_event(
                        "tool_error", {"name": name, "error": str(e)}, run_id=run_id, agent_name=agent_name
                    )
                    tool_calls.append({"name": name, "args": args, "error": str(e)})
                    results_block.append({"name": name, "ok": False, "error": str(e)})

                last_results_block = results_block

                prompt = self._build_prompt(
                    include_tools=False,
                    tools_desc=tools_desc,
                    message=message,
                    results_block=results_block,
                    step=step,
                )
                continue



 
            out = raw.strip()
            self.db.log_event(
                "final_unknown_type",
                {"output": out, "parsed": parsed},
                run_id=run_id,
                agent_name=agent_name,
            )
            return run_id, out, tool_calls

        out = self._force_finalize_from_last_tool(tool_calls, last_results_block)
        self.db.log_event("final_step_limit", {"output": out}, run_id=run_id, agent_name=agent_name)
        return run_id, out, tool_calls

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
    def _normalize_final_output(out_raw: Any, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Models sometimes return:
          - a dict {"ts": "..."} (good)
          - a JSON string "{\"ts\":\"...\"}" (annoying)
          - a python-ish string "{'ts': '...'}" (annoying)
        Normalize to a clean user-facing string.
        """

        if isinstance(out_raw, dict):
            if "ts" in out_raw:
                return str(out_raw["ts"])
            return json.dumps(out_raw, ensure_ascii=False)

        if isinstance(out_raw, list):
            return json.dumps(out_raw, ensure_ascii=False)


        s = str(out_raw).strip()
        if not s:
            return ToolCallingAgent._force_finalize_from_last_tool(tool_calls)

        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "ts" in obj:
                    return str(obj["ts"])
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass

  
        if "ts" in s and "{" in s and "}" in s
            forced = ToolCallingAgent._force_finalize_from_last_tool(tool_calls)
            if forced and forced != "I couldn't complete the request.":
                return forced

        return s

    @staticmethod
    def _force_finalize_from_last_tool(
            tool_calls: List[Dict[str, Any]],
            last_results_block: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if tool_calls:
            last = tool_calls[-1]
            res = last.get("result")

            if isinstance(res, dict) and "ts" in res:
                return str(res["ts"])

            if res is not None:
                return json.dumps(res, ensure_ascii=False)

        if last_results_block:
            return json.dumps(last_results_block, ensure_ascii=False)

        return "I couldn't complete the request."
