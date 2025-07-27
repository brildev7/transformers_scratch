"""
Multi-GPU Pretraining script for Korean sLLM
한국어 sLLM 멀티 GPU 사전학습 스크립트
"""
import os
import time
import math
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import KoreanSLLM
from .tokenizer import KoreanEnglishTokenizer
from .dataset import DatasetManager
from .config import ModelConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """멀티 GPU 학습 관리 클래스"""
    
    def __init__(self, config: ModelConfig, local_rank: int = -1):
        self.config = config
        self.local_rank = local_rank
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # 멀티 GPU 초기화
        self._setup_distributed()
        
        # 모델 및 옵티마이저 초기화
        self._setup_model()
        self._setup_optimizer()
        
        # 출력 디렉토리 생성
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 로깅 초기화 (메인 프로세스에서만)
        if self.local_rank <= 0:
            self._setup_logging()
    
    def _setup_distributed(self):
        """분산 학습 초기화"""
        if self.local_rank == -1:
            # 단일 GPU 모드
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.world_size = 1
            self.rank = 0
            logger.info(f"단일 GPU 모드, 디바이스: {self.device}")
        else:
            # 멀티 GPU 모드
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            
            # 분산 초기화
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            
            logger.info(f"멀티 GPU 모드 - Rank: {self.rank}, World Size: {self.world_size}, Local Rank: {self.local_rank}")
    
    def _setup_model(self):
        """모델 초기화"""
        self.model = KoreanSLLM(self.config).to(self.device)
        
        if self.local_rank != -1:
            # 멀티 GPU에서 DDP 사용
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # 파라미터 수 출력 (메인 프로세스에서만)
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"총 파라미터 수: {total_params:,}")
            logger.info(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 초기화"""
        # AdamW 옵티마이저
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 스케줄러 (코사인 어닐링 with 워밍업)
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
        
        # Mixed Precision을 위한 GradScaler
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        """코사인 어닐링 스케줄러 with 워밍업"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _setup_logging(self):
        """Weights & Biases 로깅 초기화"""
        try:
            wandb.init(
                project="korean-sllm",
                config=self.config.__dict__,
                name=f"korean-sllm-{time.strftime('%Y%m%d-%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"W&B 초기화 실패: {e}")
    
    def save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """체크포인트 저장"""
        if self.rank != 0:
            return
        
        # 모델 상태 가져오기
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'step': step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 일반 체크포인트 저장
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"최고 성능 모델 저장: {best_path}")
        
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """체크포인트 로드"""
        logger.info(f"체크포인트 로드: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 로드
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 로드
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['step']
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """단일 학습 스텝"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        self.model.train()
        
        if self.config.fp16:
            # Mixed Precision 학습
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            # Gradient scaling으로 역전파
            self.scaler.scale(loss).backward()
            
            # 그래디언트 클리핑
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 옵티마이저 스텝
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Full Precision 학습
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # 메트릭 계산
        with torch.no_grad():
            # Perplexity 계산
            perplexity = torch.exp(loss).item()
            
            # 정확도 계산 (다음 토큰 예측)
            logits = outputs['logits']
            predictions = torch.argmax(logits[..., :-1, :], dim=-1)
            targets = labels[..., 1:]
            
            # 패딩 토큰 제외하고 정확도 계산
            mask = (targets != -100)
            correct = (predictions == targets) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        
        metrics = {
            'perplexity': perplexity,
            'accuracy': accuracy.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return loss.item(), metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 단계"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                
                # 정확도 계산
                logits = outputs['logits']
                predictions = torch.argmax(logits[..., :-1, :], dim=-1)
                targets = labels[..., 1:]
                
                mask = (targets != -100)
                correct = (predictions == targets) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        perplexity = math.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy,
            'val_perplexity': perplexity
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: Optional[str] = None):
        """메인 학습 루프"""
        start_step = 0
        best_val_loss = float('inf')
        
        # 체크포인트에서 재시작
        if resume_from:
            start_step = self.load_checkpoint(resume_from)
        
        logger.info("사전학습 시작...")
        logger.info(f"시작 스텝: {start_step}")
        logger.info(f"최대 스텝: {self.config.max_steps}")
        
        step = start_step
        accumulation_step = 0
        
        for epoch in range(1000):  # 충분히 큰 수
            if self.local_rank != -1:
                # 분산 학습에서 에포크마다 샘플러 셔플
                train_loader.sampler.set_epoch(epoch)
            
            for batch in train_loader:
                step += 1
                
                # 학습 스텝
                loss, metrics = self.train_step(batch)
                accumulation_step += 1
                
                # 그래디언트 누적
                if accumulation_step % self.config.grad_accumulation_steps == 0:
                    accumulation_step = 0
                
                # 로깅 (메인 프로세스에서만)
                if step % 100 == 0 and self.rank == 0:
                    logger.info(
                        f"Step {step}/{self.config.max_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | "
                        f"Acc: {metrics['accuracy']:.4f} | "
                        f"LR: {metrics['learning_rate']:.2e}"
                    )
                    
                    # W&B 로깅
                    if wandb.run:
                        wandb.log({
                            'train_loss': loss,
                            'train_perplexity': metrics['perplexity'],
                            'train_accuracy': metrics['accuracy'],
                            'learning_rate': metrics['learning_rate'],
                            'step': step
                        })
                
                # 검증 및 저장
                if step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    
                    if self.rank == 0:
                        logger.info(
                            f"검증 결과 - Step {step} | "
                            f"Val Loss: {val_metrics['val_loss']:.4f} | "
                            f"Val PPL: {val_metrics['val_perplexity']:.2f} | "
                            f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                        )
                        
                        # W&B 로깅
                        if wandb.run:
                            wandb.log(val_metrics)
                        
                        # 최고 성능 모델 체크
                        is_best = val_metrics['val_loss'] < best_val_loss
                        if is_best:
                            best_val_loss = val_metrics['val_loss']
                        
                        # 체크포인트 저장
                        if step % self.config.save_steps == 0:
                            self.save_checkpoint(step, val_metrics['val_loss'], is_best)
                
                # 최대 스텝 도달 시 종료
                if step >= self.config.max_steps:
                    logger.info("최대 스텝에 도달했습니다. 학습을 종료합니다.")
                    return
        
        logger.info("학습이 완료되었습니다.")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Korean sLLM Pretraining")
    parser.add_argument("--config", type=str, default=None, help="설정 파일 경로")
    parser.add_argument("--resume_from", type=str, default=None, help="재시작할 체크포인트 경로")
    parser.add_argument("--local_rank", type=int, default=-1, help="로컬 랭크 (DDP용)")
    parser.add_argument("--download_fresh", action="store_true", help="새로운 데이터셋 다운로드")
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig()
    
    # 로컬 랭크 설정
    if args.local_rank != -1:
        config.local_rank = args.local_rank
    
    # 토크나이저 준비
    tokenizer = KoreanEnglishTokenizer(config.vocab_size)
    
    # 토크나이저가 없으면 학습
    tokenizer_path = Path(config.tokenizer_path)
    if not tokenizer_path.exists():
        logger.info("토크나이저를 학습합니다...")
        
        # 토크나이저 학습용 데이터 준비
        dataset_manager = DatasetManager(config, None)
        training_texts = dataset_manager.prepare_tokenizer_training_data()
        
        # 토크나이저 학습
        tokenizer.train(training_texts, str(tokenizer_path))
    else:
        # 기존 토크나이저 로드
        tokenizer.load(str(tokenizer_path))
    
    # 데이터셋 준비
    dataset_manager = DatasetManager(config, tokenizer)
    train_loader, val_loader = dataset_manager.prepare_datasets(args.download_fresh)
    
    # 멀티 GPU를 위한 DistributedSampler
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_loader.dataset)
        val_sampler = DistributedSampler(val_loader.dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
    
    # 트레이너 초기화 및 학습 시작
    trainer = Trainer(config, args.local_rank)
    trainer.train(train_loader, val_loader, args.resume_from)


if __name__ == "__main__":
    main() 