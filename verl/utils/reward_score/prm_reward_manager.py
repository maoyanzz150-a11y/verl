"""
PRM Reward Manager for DAPO
Integrates PRM scorer into the DAPO reward computation pipeline
"""

import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)


class PRMRewardManager:
    """
    Manages PRM scoring and integration with outcome rewards
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PRM Reward Manager
        
        Args:
            config: Configuration dict containing PRM settings
                - use_prm: bool, whether to enable PRM scoring
                - prm_model_name: str, PRM model path/name
                - prm_device: str, device to run PRM on
                - prm_mock_mode: bool, whether to use mock mode
                - prm_aggregation_method: str, how to aggregate step scores
        """
        self.enabled = config.get('use_prm', False)
        
        if not self.enabled:
            logger.info("[PRMRewardManager] PRM is disabled")
            return
        
        # Import PRM scorer here to avoid import errors if not used
        try:
            from verl.utils.reward_score.prm_scorer import PRMScorer
        except ImportError:
            logger.error("Failed to import PRMScorer. Make sure prm_scorer.py exists in verl/utils/reward_score/")
            raise
        
        # Initialize PRM scorer
        try:
            self.prm_scorer = PRMScorer(
                model_name=config.get('prm_model_name', 'Qwen/Qwen2.5-Math-PRM-7B'),
                device=config.get('prm_device', 'cuda'),
                mock_mode=config.get('prm_mock_mode', False),
                aggregation_method=config.get('prm_aggregation_method', 'mean'),
            )
            logger.info(f"[PRMRewardManager] PRM scorer initialized successfully")
        except Exception as e:
            logger.error(f"[PRMRewardManager] Failed to initialize PRM scorer: {e}")
            raise
        
        # Store configuration
        self.config = config
    
    def score_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a batch of samples with PRM
        
        Args:
            batch: Batch dict containing 'prompts' and 'responses'
                   May also contain 'group_ids' for group-wise operations
        
        Returns:
            batch: Updated batch with added PRM-related fields:
                - prm_traj_scores: List[float], trajectory-level PRM scores
                - prm_step_scores_list: List[List[float]], per-step scores
                - prm_group_ids: List[int], group IDs (same as input)
        """
        if not self.enabled:
            return batch
        
        # Extract prompts and responses from batch
        # The batch structure depends on verl's internal format
        prompts = batch.get('prompts', [])
        responses = batch.get('responses', [])
        
        if not prompts or not responses:
            logger.warning("[PRMRewardManager] Missing prompts or responses in batch")
            return batch
        
        if len(prompts) != len(responses):
            logger.error(f"Mismatch: {len(prompts)} prompts vs {len(responses)} responses")
            return batch
        
        # Score the batch
        try:
            prm_result = self.prm_scorer.score_batch(prompts, responses)
        except Exception as e:
            logger.error(f"[PRMRewardManager] PRM scoring failed: {e}")
            # Fallback: return empty scores
            batch['prm_traj_scores'] = [0.0] * len(prompts)
            batch['prm_step_scores_list'] = [[] for _ in prompts]
            return batch
        
        # Extract results
        traj_scores = prm_result['trajectory_scores']
        step_scores_list = prm_result['step_scores_list']
        
        # Add to batch
        batch['prm_traj_scores'] = traj_scores
        batch['prm_step_scores_list'] = step_scores_list
        
        # Optionally: compute high/low flags within groups
        # (This will be used by Quadrant Weighter in the next module)
        if 'group_ids' in batch:
            batch['prm_group_ids'] = batch['group_ids']
        
        logger.debug(f"[PRMRewardManager] Scored {len(traj_scores)} samples with PRM")
        
        return batch
    
    def get_stats(self, traj_scores: List[float]) -> Dict[str, float]:
        """
        Get statistics about PRM scores (for monitoring)
        
        Args:
            traj_scores: List of trajectory scores
        
        Returns:
            Dict with min, max, mean, std
        """
        if not traj_scores:
            return {}
        
        scores_arr = np.array(traj_scores)
        return {
            'prm_min': float(np.min(scores_arr)),
            'prm_max': float(np.max(scores_arr)),
            'prm_mean': float(np.mean(scores_arr)),
            'prm_std': float(np.std(scores_arr)),
        }
