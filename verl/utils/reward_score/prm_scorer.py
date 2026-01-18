"""
PRM (Process Reward Model) Scorer for Qwen2.5-Math-PRM-7B
Based on official Qwen implementation with production enhancements
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def make_step_rewards(logits, token_masks):
    """
    Official Qwen implementation for extracting step-level rewards
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    
    return all_scores_res


class PRMScorer:
    """Process Reward Model Scorer for DAPO/GRPO training"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: str = "auto",
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
        step_split_method: str = "double_newline",
        aggregation_method: str = "mean",
        score_clipping: bool = True,
        clip_epsilon: float = 1e-4,
        mock_mode: bool = False,
        mock_seed: int = 42,
    ):
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt
        self.step_split_method = step_split_method
        self.aggregation_method = aggregation_method
        self.score_clipping = score_clipping
        self.clip_epsilon = clip_epsilon
        self.mock_mode = mock_mode
        self.model = None
        self.tokenizer = None
        self.step_sep_id = None
        
        if mock_mode:
            np.random.seed(mock_seed)
            logger.info("[PRMScorer] Initialized in MOCK mode")
            return
        
        logger.info(f"[PRMScorer] Loading model from {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
            
            # ✅ Bug 修复 3: 添加 add_special_tokens=False
            extra_0_ids = self.tokenizer.encode(
                "<extra_0>",
                add_special_tokens=False  # 关键！防止 token ID 偏移
            )
            if len(extra_0_ids) > 0:
                self.step_sep_id = extra_0_ids[0]
                logger.info(f"[PRMScorer] Model loaded. <extra_0> token ID: {self.step_sep_id}")
            else:
                raise ValueError("Failed to find <extra_0> token ID in tokenizer")
                
        except Exception as e:
            logger.error(f"[PRMScorer] Failed to load model: {e}")
            raise
    
    def split_steps(self, response_text: str) -> List[str]:
        """Split response text into steps"""
        if not response_text or not response_text.strip():
            logger.warning("[PRMScorer] Empty response text")
            return []
        
        if self.step_split_method == "double_newline":
            steps = response_text.split("\n\n")
        elif self.step_split_method == "single_newline":
            steps = response_text.split("\n")
        else:
            raise ValueError(f"Unknown step_split_method: {self.step_split_method}")
        
        steps = [s.strip() for s in steps if s.strip()]
        
        if not steps:
            logger.warning("[PRMScorer] No steps found after splitting")
            return []
        
        return steps
    
    def _get_step_rewards_internal(
        self,
        prompt: str,
        response_text: str,
    ) -> Tuple[List[float], int, int]:
        """Internal implementation: compute step rewards using model inference"""
        steps = self.split_steps(response_text)
        num_steps_split = len(steps)
        
        if num_steps_split == 0:
            logger.warning("[PRMScorer] No steps to score")
            return [], 0, 0
        
        assistant_content = "<extra_0>".join(steps) + "<extra_0>"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
        
        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # ✅ Bug 修复 1: 改用 tokenizer() 而不是 tokenizer.encode()
        input_tokens = self.tokenizer(
            conversation_str,
            return_tensors="pt",
        )
        input_ids = input_tokens["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        
        # ✅ Bug 修复 2: 优先用 outputs.logits，再 fallback 到 outputs[0]
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]
        
        token_masks = (input_ids == self.step_sep_id)
        batch_step_rewards = make_step_rewards(logits, token_masks)
        
        step_rewards = batch_step_rewards[0] if batch_step_rewards else []
        num_steps_scored = len(step_rewards)
        
        if num_steps_scored != num_steps_split:
            logger.warning(
                f"[PRMScorer] Step count mismatch: split={num_steps_split}, "
                f"scored={num_steps_scored}. Taking min({num_steps_split}, {num_steps_scored}) steps."
            )
            min_len = min(num_steps_scored, num_steps_split)
            step_rewards = step_rewards[:min_len]
            num_steps_scored = min_len
        
        return step_rewards, num_steps_split, num_steps_scored
    
    def _get_step_rewards_mock(self, response_text: str) -> Tuple[List[float], int, int]:
        """Mock implementation with more realistic score distribution"""
        steps = self.split_steps(response_text)
        num_steps = len(steps)
        
        # ✅ 改进：70% 高分，30% 低分，更接近真实 PRM
        mock_scores = []
        for i in range(num_steps):
            if np.random.random() < 0.7:
                # 正常高分
                score = np.clip(np.random.normal(0.7, 0.15), 0.1, 0.99)
            else:
                # 异常低分（用于测试 C 类：错误但 PRM 高）
                score = np.random.uniform(0.05, 0.4)
            mock_scores.append(score)
        
        return mock_scores, num_steps, num_steps
    
    def get_step_scores(
        self,
        prompt: str,
        response_text: str,
    ) -> Dict[str, any]:
        """Compute step-level scores for a prompt-response pair"""
        
        if self.mock_mode:
            step_scores, num_steps_split, num_steps_scored = self._get_step_rewards_mock(response_text)
        else:
            step_scores, num_steps_split, num_steps_scored = self._get_step_rewards_internal(
                prompt, response_text
            )
        
        # ✅ 新增：Score clipping 保证数值稳定性
        if self.score_clipping and step_scores:
            step_scores = [
                max(self.clip_epsilon, min(score, 1.0 - self.clip_epsilon))
                for score in step_scores
            ]
        
        traj_score = self.aggregate_scores(step_scores)
        
        return {
            "step_scores": step_scores,
            "num_steps": len(step_scores),
            "traj_score": traj_score,
            "debug_info": {
                "num_steps_split": num_steps_split,
                "num_steps_scored": num_steps_scored,
                "aggregation_method": self.aggregation_method,
            }
        }
    
    def aggregate_scores(self, step_scores: List[float]) -> float:
        """Aggregate step-level scores into a single trajectory score"""
        if not step_scores:
            logger.warning("[PRMScorer] Empty step_scores in aggregation")
            return 0.0
        
        step_scores_arr = np.array(step_scores)
        
        if self.aggregation_method == "mean":
            return float(np.mean(step_scores_arr))
        elif self.aggregation_method == "min":
            return float(np.min(step_scores_arr))
        elif self.aggregation_method == "product":
            return float(np.prod(step_scores_arr))
        else:
            logger.warning(f"Unknown aggregation method: {self.aggregation_method}, using mean")
            return float(np.mean(step_scores_arr))
    
    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, any]:
        """Score a batch of (prompt, response) pairs"""
        trajectory_scores = []
        step_scores_list = []
        batch_debug_info = []
        
        for prompt, response in zip(prompts, responses):
            result = self.get_step_scores(prompt, response)
            trajectory_scores.append(result["traj_score"])
            step_scores_list.append(result["step_scores"])
            batch_debug_info.append(result["debug_info"])
        
        return {
            "trajectory_scores": trajectory_scores,
            "step_scores_list": step_scores_list,
            "batch_debug_info": batch_debug_info,
        }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试 Mock 模式
    print("=" * 80)
    print("TEST: Mock Mode")
    print("=" * 80)
    scorer = PRMScorer(mock_mode=True)
    
    prompt = "Sue lives in a fun neighborhood..."
    response = """First step here.

Second step here.

Third step here.

Final answer here."""
    
    result = scorer.get_step_scores(prompt, response)
    print(f"Step Scores: {result['step_scores']}")
    print(f"Trajectory Score: {result['traj_score']:.4f}")
    print(f"Num Steps: {result['num_steps']}")
    print(f"Debug Info: {result['debug_info']}")
