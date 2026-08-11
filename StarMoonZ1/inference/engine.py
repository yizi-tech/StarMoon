# 推理引擎 - 支持 PyTorch / vLLM / llama.cpp 多后端 + WebUI

from __future__ import annotations
import os, json, time, logging, threading
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from StarMoonZ1.model.config import StarMoonZ1Config
from StarMoonZ1.model.model import StarMoonZ1ForCausalLM

logger = logging.getLogger("StarMoonZ1.Inference")

WEBUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: int = 2


class InferenceEngine:
    def __init__(self, model_path: str, backend: str = "auto",
                 torch_dtype=torch.bfloat16, device_map: str = "auto",
                 use_flash_attn: bool = True):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()  # 保护并发推理

        if backend == "auto":
            backend = self._detect_backend(model_path)
        self.backend = backend

        logger.info(f"Using inference backend: {backend}")
        self._load(model_path, torch_dtype, device_map, use_flash_attn)

    def _detect_backend(self, path):
        if path.endswith(".gguf"):
            return "llamacpp"
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return "transformers"
        if "/" in path:
            return "transformers"
        return "transformers"

    def _load(self, path, dtype, device_map, flash):
        if self.backend == "transformers":
            self.model = StarMoonZ1ForCausalLM.from_pretrained(
                path, torch_dtype=dtype, device_map=device_map, use_flash_attn=flash,
            )
        elif self.backend == "vllm":
            from vllm import LLM
            self.model = LLM(model=path, dtype=str(dtype).split(".")[-1], trust_remote_code=True)
        elif self.backend == "llamacpp":
            from llama_cpp import Llama
            self.model = Llama(model_path=path, n_ctx=32768, verbose=False)

    def load_tokenizer(self, path=None):
        from transformers import AutoTokenizer
        path = path or self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> str:
        if gen_config is None:
            gen_config = GenerationConfig()

        with self._lock:
            if self.backend == "transformers":
                return self._generate_hf(prompt, gen_config)
            elif self.backend == "vllm":
                return self._generate_vllm(prompt, gen_config)
            elif self.backend == "llamacpp":
                return self._generate_llamacpp(prompt, gen_config)

    def _generate_hf(self, prompt, gc):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.model.token_embedding.weight.device
        )
        out_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=gc.max_new_tokens,
            temperature=gc.temperature, top_p=gc.top_p, top_k=gc.top_k,
            repetition_penalty=gc.repetition_penalty,
            eos_token_id=gc.eos_token_id, do_sample=gc.do_sample,
        )
        return self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _generate_vllm(self, prompt, gc):
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=gc.temperature, top_p=gc.top_p, top_k=gc.top_k,
            max_tokens=gc.max_new_tokens, repetition_penalty=gc.repetition_penalty,
        )
        outputs = self.model.generate([prompt], params)
        return outputs[0].outputs[0].text

    def _generate_llamacpp(self, prompt, gc):
        outputs = self.model(
            prompt, max_tokens=gc.max_new_tokens,
            temperature=gc.temperature, top_p=gc.top_p, top_k=gc.top_k,
            repeat_penalty=gc.repetition_penalty, echo=False,
        )
        return outputs["choices"][0]["text"]

    def chat(self, messages: List[Dict[str, str]], gc: Optional[GenerationConfig] = None) -> str:
        if self.tokenizer is None:
            self.load_tokenizer()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return self.generate(prompt, gc)

    # ------------------------------------------------------------------
    # 流式生成
    # ------------------------------------------------------------------
    def generate_stream(self, prompt: str, gc: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        """逐 token 流式生成，yield 增量文本片段"""
        if gc is None:
            gc = GenerationConfig()
        if self.backend == "transformers":
            yield from self._stream_hf(prompt, gc)
        else:
            # 非 transformers 后端回退为一次性生成
            yield self.generate(prompt, gc)

    def chat_stream(self, messages: List[Dict[str, str]], gc: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        """多轮对话流式生成"""
        if self.tokenizer is None:
            self.load_tokenizer()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        yield from self.generate_stream(prompt, gc)

    def _stream_hf(self, prompt: str, gc: GenerationConfig) -> Generator[str, None, None]:
        """transformers 后端逐 token 流式推理 (KV Cache 增量解码)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.model.token_embedding.weight.device
        )
        input_ids = inputs["input_ids"]
        self.model.eval()

        pkv, gen = None, input_ids
        emitted = ""  # 已输出的完整文本

        with torch.no_grad():
            for _ in range(gc.max_new_tokens):
                mi = gen[:, -1:] if pkv is not None else gen
                # KV-cache 增量解码时传入新 token 绝对位置，确保 RoPE 正确
                pos_ids = None
                if pkv is not None:
                    pos_ids = torch.full((gen.shape[0], 1), gen.shape[1] - 1,
                                         dtype=torch.long, device=gen.device)
                o = self.model(mi, None, None, pkv, True, position_ids=pos_ids)
                pkv = o["past_key_values"]
                nl = o["logits"][:, -1, :] / max(gc.temperature, 1e-8)

                if gc.repetition_penalty != 1.0:
                    penalty_mask = torch.zeros_like(nl)
                    penalty_mask.scatter_(1, gen, 1.0)
                    pen_mask = penalty_mask.bool()
                    scale = torch.where(nl > 0, 1.0 / gc.repetition_penalty, gc.repetition_penalty)
                    nl = torch.where(pen_mask, nl * scale, nl)
                if gc.top_k > 0:
                    kv, _ = torch.topk(nl, gc.top_k, dim=-1)
                    nl[nl < kv[:, -1, None]] = float("-inf")
                if gc.top_p < 1.0:
                    sl, si = torch.sort(nl, descending=True, dim=-1)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cp > gc.top_p
                    rm[:, 1:] = rm[:, :-1].clone()
                    rm[:, 0] = False
                    nl[rm.scatter(1, si, rm)] = float("-inf")

                if gc.do_sample:
                    nt = torch.multinomial(F.softmax(nl, dim=-1), 1)
                else:
                    nt = nl.argmax(dim=-1, keepdim=True)

                gen = torch.cat([gen, nt], dim=-1)
                if (nt == gc.eos_token_id).all():
                    break

                # 增量解码: 解码全部已生成 token，输出新增部分
                full_text = self.tokenizer.decode(
                    gen[0][input_ids.shape[1]:], skip_special_tokens=True)
                if len(full_text) > len(emitted):
                    delta = full_text[len(emitted):]
                    emitted = full_text
                    yield delta

        # 输出剩余未发送部分
        final_text = self.tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        if len(final_text) > len(emitted):
            yield final_text[len(emitted):]

    def create_server(self, host="0.0.0.0", port=8000, webui: bool = True,
                      cors_origins: Optional[List[str]] = None):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
        import uvicorn

        app = FastAPI(title="StarMoon-z1", version="0.1.0")
        # CORS: 生产环境应明确指定允许的源，而非通配符
        allowed_origins = cors_origins or ["http://localhost:3000", "http://127.0.0.1:3000"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"],
        )

        class GenerateReq(BaseModel):
            prompt: str = Field(..., min_length=1, max_length=100000)
            max_new_tokens: int = Field(default=1024, ge=1, le=8192)
            temperature: float = Field(default=0.7, ge=0.0, le=2.0)
            top_p: float = Field(default=0.9, ge=0.0, le=1.0)
            stream: bool = False

        class ChatReq(BaseModel):
            messages: List[Dict[str, str]] = Field(..., min_length=1, max_length=100)
            max_new_tokens: int = Field(default=2048, ge=1, le=8192)
            temperature: float = Field(default=0.7, ge=0.0, le=2.0)
            top_p: float = Field(default=0.9, ge=0.0, le=1.0)
            top_k: int = Field(default=50, ge=0, le=200)
            repetition_penalty: float = Field(default=1.0, ge=0.5, le=3.0)
            stream: bool = True

        @app.post("/generate")
        def gen_endpoint(req: GenerateReq):
            gc = GenerationConfig(max_new_tokens=req.max_new_tokens,
                                  temperature=req.temperature, top_p=req.top_p)
            if req.stream:
                def sse():
                    for chunk in self.generate_stream(req.prompt, gc):
                        yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(sse(), media_type="text/event-stream")
            return {"response": self.generate(req.prompt, gc)}

        @app.post("/chat")
        def chat_endpoint(req: ChatReq):
            gc = GenerationConfig(
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_p=req.top_p,
                top_k=req.top_k, repetition_penalty=req.repetition_penalty,
                do_sample=req.temperature > 0,
            )
            if req.stream:
                def sse():
                    t0 = time.time()
                    tokens = 0
                    for chunk in self.chat_stream(req.messages, gc):
                        tokens += 1
                        yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                    elapsed = time.time() - t0
                    yield f"data: {json.dumps({'done': True, 'tokens': tokens, 'time': round(elapsed, 2)})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(sse(), media_type="text/event-stream")
            t0 = time.time()
            response = self.chat(req.messages, gc)
            return {"response": response, "time": round(time.time() - t0, 2)}

        @app.get("/health")
        def health():
            return {
                "status": "ok", "backend": self.backend,
                "model": self.model_path,
                "webui": webui and os.path.isdir(WEBUI_DIR),
            }

        # WebUI 静态文件
        if webui and os.path.isdir(WEBUI_DIR):
            app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")

            @app.get("/")
            def index():
                return FileResponse(os.path.join(WEBUI_DIR, "index.html"))

            logger.info(f"WebUI enabled: http://{host}:{port}/")

        logger.info(f"Server starting on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
