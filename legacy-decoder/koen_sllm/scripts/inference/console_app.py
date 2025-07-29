"""
한국어 소형 언어모델 대화형 콘솔 애플리케이션
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Optional
from datetime import datetime
import readline  # 명령어 히스토리 지원

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from inference_engine import InferenceEngine


class ConsoleApp:
    """대화형 콘솔 애플리케이션"""
    
    def __init__(self, engine: InferenceEngine):
        """
        Args:
            engine: 추론 엔진 인스턴스
        """
        self.engine = engine
        self.chat_history: List[Dict[str, str]] = []
        self.session_start_time = datetime.now()
        
        # 설정
        # 설정 (텍스트 생성 최적화)
        self.max_length = 150
        self.temperature = 1.5  # 더 다양한 토큰 선택을 위해 상향 조정
        self.top_k = 50
        self.top_p = 0.95  # 더 많은 토큰이 선택되도록 상향 조정
        self.do_sample = True  # 샘플링 방식으로 변경
        
        print("=" * 60)
        print("🤖 한국어 소형 언어모델 대화형 콘솔")
        print("=" * 60)
        print()
        
        # 모델 정보 출력
        model_info = self.engine.get_model_info()
        print("📊 모델 정보:")
        print(f"  • 모델명: {model_info['model_name']}")
        print(f"  • 파라미터 수: {model_info['model_parameters']:,}")
        print(f"  • 어휘 크기: {model_info['vocab_size']:,}")
        print(f"  • 디바이스: {model_info['device']}")
        print(f"  • 최대 시퀀스 길이: {model_info['max_position_embeddings']}")
        print()
        
        # 명령어 도움말
        self.show_help()
    
    def show_help(self):
        """도움말 출력"""
        print("💡 사용 방법:")
        print("  • 일반 대화: 텍스트를 입력하세요")
        print("  • /help         - 도움말 표시")
        print("  • /info         - 모델 정보 표시")
        print("  • /settings     - 현재 설정 표시")
        print("  • /temp <값>    - 온도 설정 (0.1-2.0)")
        print("  • /length <값>  - 최대 생성 길이 설정")
        print("  • /clear        - 대화 히스토리 초기화")
        print("  • /save <파일명> - 대화 히스토리 저장")
        print("  • /load <파일명> - 대화 히스토리 불러오기")
        print("  • /benchmark    - 성능 테스트 실행")
        print("  • /complete     - 텍스트 완성 모드")
        print("  • /multiple <n> - 다중 응답 생성 (n개)")
        print("  • /exit 또는 /quit - 종료")
        print()
    
    def show_settings(self):
        """현재 설정 표시"""
        print("⚙️ 현재 설정:")
        print(f"  • 최대 길이: {self.max_length}")
        print(f"  • 온도: {self.temperature}")
        print(f"  • Top-k: {self.top_k}")
        print(f"  • Top-p: {self.top_p}")
        print(f"  • 샘플링: {self.do_sample}")
        print()
    
    def save_history(self, filename: str):
        """대화 히스토리 저장"""
        try:
            history_data = {
                "session_start": self.session_start_time.isoformat(),
                "settings": {
                    "max_length": self.max_length,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p
                },
                "chat_history": self.chat_history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 대화 히스토리가 저장되었습니다: {filename}")
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    def load_history(self, filename: str):
        """대화 히스토리 불러오기"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.chat_history = history_data.get("chat_history", [])
            
            # 설정도 불러오기 (선택적)
            settings = history_data.get("settings", {})
            self.max_length = settings.get("max_length", self.max_length)
            self.temperature = settings.get("temperature", self.temperature)
            self.top_k = settings.get("top_k", self.top_k)
            self.top_p = settings.get("top_p", self.top_p)
            
            print(f"✅ 대화 히스토리가 불러와졌습니다: {filename}")
            print(f"   불러온 대화 수: {len(self.chat_history)}")
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {filename}")
        except Exception as e:
            print(f"❌ 불러오기 실패: {e}")
    
    def complete_text_mode(self):
        """텍스트 완성 모드"""
        print("\n📝 텍스트 완성 모드 (빈 줄 입력 시 종료)")
        print("불완전한 텍스트를 입력하면 모델이 완성해드립니다.")
        
        while True:
            try:
                incomplete_text = input("\n완성할 텍스트: ").strip()
                
                if not incomplete_text:
                    print("텍스트 완성 모드를 종료합니다.")
                    break
                
                completed = self.engine.complete_text(
                    incomplete_text=incomplete_text,
                    max_completion_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
                
                print(f"\n✨ 완성된 텍스트:\n{completed}")
                
            except KeyboardInterrupt:
                print("\n텍스트 완성 모드를 종료합니다.")
                break
    
    def multiple_responses_mode(self, num_responses: int):
        """다중 응답 생성 모드"""
        print(f"\n🔀 다중 응답 모드 ({num_responses}개 응답 생성)")
        
        try:
            prompt = input("프롬프트: ").strip()
            
            if not prompt:
                print("빈 프롬프트입니다.")
                return
            
            responses = self.engine.generate_text(
                prompt=prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_return_sequences=num_responses,
                return_prompt=False
            )
            
            print(f"\n✨ {num_responses}개의 응답:")
            for i, response in enumerate(responses, 1):
                print(f"\n[응답 {i}] {response}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def process_command(self, user_input: str) -> bool:
        """명령어 처리
        
        Returns:
            계속 실행할지 여부 (False면 종료)
        """
        command_parts = user_input.split()
        command = command_parts[0].lower()
        
        if command in ["/exit", "/quit"]:
            return False
            
        elif command == "/help":
            self.show_help()
            
        elif command == "/info":
            model_info = self.engine.get_model_info()
            print("\n📊 모델 정보:")
            for key, value in model_info.items():
                print(f"  • {key}: {value}")
            print()
            
        elif command == "/settings":
            self.show_settings()
            
        elif command == "/temp":
            if len(command_parts) > 1:
                try:
                    new_temp = float(command_parts[1])
                    if 0.1 <= new_temp <= 2.0:
                        self.temperature = new_temp
                        print(f"✅ 온도가 {new_temp}로 설정되었습니다.")
                    else:
                        print("❌ 온도는 0.1과 2.0 사이여야 합니다.")
                except ValueError:
                    print("❌ 올바른 숫자를 입력하세요.")
            else:
                print("❌ 사용법: /temp <값>")
                
        elif command == "/length":
            if len(command_parts) > 1:
                try:
                    new_length = int(command_parts[1])
                    if 10 <= new_length <= 1000:
                        self.max_length = new_length
                        print(f"✅ 최대 길이가 {new_length}로 설정되었습니다.")
                    else:
                        print("❌ 길이는 10과 1000 사이여야 합니다.")
                except ValueError:
                    print("❌ 올바른 숫자를 입력하세요.")
            else:
                print("❌ 사용법: /length <값>")
                
        elif command == "/clear":
            self.chat_history.clear()
            print("✅ 대화 히스토리가 초기화되었습니다.")
            
        elif command == "/save":
            if len(command_parts) > 1:
                filename = command_parts[1]
                if not filename.endswith('.json'):
                    filename += '.json'
                self.save_history(filename)
            else:
                # 기본 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.json"
                self.save_history(filename)
                
        elif command == "/load":
            if len(command_parts) > 1:
                filename = command_parts[1]
                if not filename.endswith('.json'):
                    filename += '.json'
                self.load_history(filename)
            else:
                print("❌ 사용법: /load <파일명>")
                
        elif command == "/benchmark":
            print("\n🏃 성능 벤치마크 실행 중...")
            results = self.engine.benchmark()
            print()
            
        elif command == "/complete":
            self.complete_text_mode()
            
        elif command == "/multiple":
            if len(command_parts) > 1:
                try:
                    num_responses = int(command_parts[1])
                    if 1 <= num_responses <= 10:
                        self.multiple_responses_mode(num_responses)
                    else:
                        print("❌ 응답 수는 1과 10 사이여야 합니다.")
                except ValueError:
                    print("❌ 올바른 숫자를 입력하세요.")
            else:
                print("❌ 사용법: /multiple <응답수>")
                
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print("   /help를 입력하여 사용 가능한 명령어를 확인하세요.")
        
        return True
    
    def run(self):
        """메인 루프 실행"""
        print("💬 대화를 시작하세요! (/help로 명령어 확인)")
        print("-" * 60)
        
        try:
            while True:
                try:
                    # 사용자 입력 받기
                    user_input = input("\n🧑 사용자: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # 명령어 처리
                    if user_input.startswith('/'):
                        should_continue = self.process_command(user_input)
                        if not should_continue:
                            break
                        continue
                    
                    # 일반 대화 처리
                    print("\n🤖 모델: ", end="", flush=True)
                    
                    
                    response = self.engine.chat_generate(
                        message=user_input,
                        chat_history=self.chat_history,
                        max_length=self.max_length,
                        temperature=self.temperature
                    )
                    
                    
                    print(response)
                    
                    # 히스토리에 추가
                    self.chat_history.append({"role": "user", "content": user_input})
                    self.chat_history.append({"role": "assistant", "content": response})
                    
                    # 히스토리 크기 제한 (메모리 관리)
                    if len(self.chat_history) > 100:
                        self.chat_history = self.chat_history[-80:]  # 최근 80개 대화만 유지
                    
                except KeyboardInterrupt:
                    print("\n\n🔄 입력을 취소했습니다. 계속하려면 텍스트를 입력하거나 /exit로 종료하세요.")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        
        # 세션 정보 출력
        session_duration = datetime.now() - self.session_start_time
        print(f"\n📊 세션 정보:")
        print(f"  • 지속 시간: {session_duration}")
        print(f"  • 총 대화 수: {len(self.chat_history)}")
        print("\n" + "=" * 60)
        print("프로그램을 종료합니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="한국어 소형 언어모델 대화형 콘솔")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/checkpoint-12000",
        help="모델 체크포인트 경로"
    )
    
    parser.add_argument(
        "--use-legacy-model",
        action="store_true",
        help="기존 InferenceModel 사용 (기본값: 학습 호환 모델)"
    )
    parser.add_argument(
        "--use-basic-tokenizer",
        action="store_true",
        help="기본 토크나이저 사용 (기본값: 개선된 토크나이저)"
    )
    parser.add_argument(
        "--use-korean-tokenizer",
        action="store_true",
        help="한국어 토크나이저 사용 (형태소 분석 지원)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="토크나이저 어휘 파일 경로 (선택적)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스"
    )
    
    args = parser.parse_args()
    
    try:
        print("모델 로딩 중...")
        
        # 체크포인트 경로 확인
        if not os.path.exists(args.checkpoint):
            print(f"❌ 체크포인트 경로를 찾을 수 없습니다: {args.checkpoint}")
            print("사용 가능한 체크포인트:")
            outputs_dir = "./outputs"
            if os.path.exists(outputs_dir):
                checkpoints = [d for d in os.listdir(outputs_dir) if d.startswith("checkpoint-")]
                for cp in sorted(checkpoints):
                    print(f"  • {outputs_dir}/{cp}")
            sys.exit(1)
        
        # 추론 엔진 로드
        use_training_compatible = not args.use_legacy_model  # 기본값: True
        use_improved_tokenizer = not args.use_basic_tokenizer  # 기본값: True
        use_korean_tokenizer = args.use_korean_tokenizer  # 기본값: False
        
        print(f"{"🔄 학습 호환 모드" if use_training_compatible else "📦 레거시 모델 모드"} 사용")
        if use_korean_tokenizer:
            print("🇰🇷 한국어 토크나이저 사용")
        elif use_improved_tokenizer:
            print("🚀 개선된 토크나이저 사용")
        else:
            print("📝 기본 토크나이저 사용")
        
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=args.device,
            use_training_compatible=use_training_compatible,
            use_improved_tokenizer=use_improved_tokenizer,
            use_korean_tokenizer=use_korean_tokenizer
        )
        
        # 콘솔 앱 실행
        app = ConsoleApp(engine)
        app.run()
        
    except KeyboardInterrupt:
        print("\n프로그램이 취소되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 