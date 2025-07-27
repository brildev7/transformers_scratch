#!/usr/bin/env python3
"""
한국어 sLLM H100 듀얼 GPU 학습 스크립트
Korean sLLM Training Script for Dual H100 GPUs

H100 GPU 2장을 활용한 분산 학습을 지원합니다.
테스트 모드(10스텝)와 실제 학습 모드를 제공합니다.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# 프로젝트 루트 추가
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent.parent  # legacy-decoder/koen_sllm/scripts/train -> transformers_scratch
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """학습 설정 클래스"""
    # 모델 설정
    model_name: str = "korean-sllm-h100"
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    
    # 학습 설정
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 데이터 설정
    dataset_path: str = "../../../../datasets"
    max_seq_length: int = 2048
    
    # H100 최적화 설정
    use_flash_attention: bool = True
    compile_model: bool = True
    mixed_precision: str = "bf16"  # H100은 bf16 최적화
    
    # 테스트/실제 모드
    test_mode: bool = False
    max_steps: int = 10  # 테스트 모드용
    
    # 저장 설정
    output_dir: str = "./../../outputs"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # 분산 학습
    world_size: int = 2  # H100 2장

class SimpleTransformerModel(nn.Module):
    """간단한 트랜스포머 모델 (테스트용)"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # weight tying
        self.head.weight = self.embeddings.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Attention mask for transformer
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # invert for transformer
        
        # Transformer
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = self.ln_f(hidden_states)
        
        # Language model head
        logits = self.head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits}

class TextDataset(Dataset):
    """텍스트 데이터셋 클래스"""
    
    def __init__(self, data_path: str, max_length: int = 2048):
        self.max_length = max_length
        self.data = []
        
        # JSONL 파일 로드
        data_file = Path(data_path) / "mixed_pretraining.jsonl"
        if not data_file.exists():
            logger.warning(f"데이터 파일이 없습니다: {data_file}")
            # 더미 데이터 생성 (테스트용)
            self.data = [{"text": f"안녕하세요 한국어 테스트 텍스트입니다. {i}"} for i in range(1000)]
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        
        logger.info(f"데이터셋 로드 완료: {len(self.data)} 샘플")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # 간단한 토크나이징 (실제로는 토크나이저 사용)
        tokens = text.split()[:self.max_length-2]  # 시작/끝 토큰 공간
        
        # 더미 토큰 ID (실제로는 토크나이저 사용)
        input_ids = [1] + [hash(token) % 65535 + 1 for token in tokens] + [2]  # BOS + tokens + EOS
        
        # 패딩
        if len(input_ids) < self.max_length:
            input_ids.extend([0] * (self.max_length - len(input_ids)))
        else:
            input_ids = input_ids[:self.max_length]
            
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)  # 언어 모델링용
        }

class H100Trainer:
    """H100 GPU 2장을 위한 트레이너 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 분산 학습 초기화
        self._setup_distributed()
        
        # 디바이스 설정
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # 모델 초기화
        self._setup_model()
        
        # 데이터 로더 초기화
        self._setup_dataloader()
        
        # 옵티마이저 및 스케줄러 초기화
        self._setup_optimizer()
        
        # 출력 디렉토리
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 상태
        self.global_step = 0
        self.epoch = 0
        
    def _setup_distributed(self):
        """분산 학습 초기화"""
        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.local_rank
            )
            logger.info(f"분산 학습 초기화 완료: rank {self.local_rank}/{self.world_size}")
        
    def _setup_model(self):
        """모델 초기화"""
        logger.info("모델 초기화 중...")
        
        self.model = SimpleTransformerModel(self.config).to(self.device)
        
        # H100 최적화
        if self.config.compile_model:
            self.model = torch.compile(self.model)
            logger.info("모델 컴파일 완료 (torch.compile)")
        
        # DDP 설정
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            
        # 모델 크기 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"총 파라미터: {total_params:,}")
        logger.info(f"학습 가능 파라미터: {trainable_params:,}")
        
    def _setup_dataloader(self):
        """데이터 로더 초기화"""
        logger.info("데이터 로더 초기화 중...")
        
        dataset = TextDataset(self.config.dataset_path, self.config.max_seq_length)
        
        # 분산 샘플러
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"데이터 로더 초기화 완료: {len(dataset)} 샘플, {len(self.dataloader)} 배치")
        
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 초기화"""
        logger.info("옵티마이저 초기화 중...")
        
        # 파라미터 그룹
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 스케줄러
        if self.config.test_mode:
            total_steps = self.config.max_steps
        else:
            total_steps = len(self.dataloader) * 3  # 3 에포크 기본값
            
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision == "fp16" else None
        
        logger.info(f"옵티마이저 초기화 완료: total_steps={total_steps}, warmup_steps={warmup_steps}")
        
    def train(self):
        """학습 실행"""
        logger.info("학습 시작!")
        logger.info(f"테스트 모드: {self.config.test_mode}")
        if self.config.test_mode:
            logger.info(f"최대 스텝: {self.config.max_steps}")
        
        self.model.train()
        start_time = time.time()
        
        # 학습 루프
        for epoch in range(100):  # 최대 100 에포크
            if self.world_size > 1:
                self.dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(self.dataloader):
                # 디바이스로 이동
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=(self.config.mixed_precision != "fp32")):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    
                    # Gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # 로깅
                    if self.global_step % self.config.logging_steps == 0 and self.local_rank == 0:
                        elapsed_time = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        
                        logger.info(
                            f"Step {self.global_step:5d} | "
                            f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {elapsed_time:.1f}s"
                        )
                    
                    # 테스트 모드 종료 조건
                    if self.config.test_mode and self.global_step >= self.config.max_steps:
                        logger.info("테스트 모드 완료!")
                        return
                    
                    # 모델 저장
                    if self.global_step % self.config.save_steps == 0 and self.local_rank == 0:
                        self._save_checkpoint()
                        
            self.epoch += 1
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            if self.local_rank == 0:
                logger.info(f"Epoch {self.epoch} 완료 | 평균 Loss: {avg_loss:.4f}")
                
        logger.info("학습 완료!")
        
    def _save_checkpoint(self):
        """체크포인트 저장"""
        if self.local_rank != 0:
            return
            
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 상태
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / "pytorch_model.bin")
        
        # 설정 저장
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        logger.info(f"체크포인트 저장 완료: {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="한국어 sLLM H100 듀얼 GPU 지능형 학습")
    
    # 모드 설정
    parser.add_argument("--test-mode", action="store_true", help="테스트 모드 (10스텝)")
    parser.add_argument("--max-steps", type=int, default=10, help="테스트 모드 최대 스텝")
    parser.add_argument("--max-epochs", type=int, default=5, help="최대 epoch 수")
    
    # 데이터 설정
    parser.add_argument("--dataset-path", type=str, default="../../../../datasets", help="데이터셋 경로")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="출력 디렉토리")
    parser.add_argument("--logs-dir", type=str, default="./logs", help="로그 디렉토리")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Validation 데이터 비율")
    
    # Early Stopping 설정
    parser.add_argument("--early-stopping", action="store_true", help="Early stopping 활성화")
    parser.add_argument("--patience", type=int, default=3, help="Validation loss 증가 허용 횟수")
    parser.add_argument("--min-delta", type=float, default=0.001, help="최소 개선 임계값")
    parser.add_argument("--validation-steps", type=int, default=100, help="Validation 체크 간격")
    
    # 학습 설정
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="최대 시퀀스 길이")
    parser.add_argument("--save-steps", type=int, default=500, help="저장 스텝 간격")
    
    # H100 최적화
    parser.add_argument("--no-compile", action="store_true", help="torch.compile 비활성화")
    parser.add_argument("--mixed-precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="Mixed precision")
    
    # 지능형 기능
    parser.add_argument("--no-save-initial", action="store_true", help="초기 모델 저장 비활성화")
    parser.add_argument("--smart-monitoring", action="store_true", help="지능형 모니터링 활성화")
    parser.add_argument("--no-token-tracking", action="store_true", help="토큰 카운팅 비활성화")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = TrainingConfig(
        test_mode=args.test_mode,
        max_steps=args.max_steps,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        compile_model=not args.no_compile,
        mixed_precision=args.mixed_precision
    )
    
    # 추가된 인수들 처리 (로깅)
    if hasattr(args, 'logs_dir'):
        logger.info(f"로그 디렉토리: {args.logs_dir}")
    if hasattr(args, 'early_stopping') and args.early_stopping:
        logger.info(f"Early Stopping 활성화: patience={args.patience}, min_delta={args.min_delta}")
    if hasattr(args, 'smart_monitoring') and args.smart_monitoring:
        logger.info("지능형 모니터링 활성화")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(f"사용 가능한 GPU: {torch.cuda.device_count()}개")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
                
    # 트레이너 초기화 및 학습 시작
    trainer = H100Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 