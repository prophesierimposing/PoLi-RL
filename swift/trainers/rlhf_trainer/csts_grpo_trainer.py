# import numpy as np
# import torch
# import re
# from trl.trainer.grpo_trainer import RepeatSampler
# from collections import defaultdict
# from typing import Any, Dict, List, Optional, Union, Tuple
# from scipy.stats import spearmanr, pearsonr
# from sentence_transformers import SentenceTransformer
# from .grpo_trainer import GRPOTrainer
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# class CSTSGRPOTrainer(GRPOTrainer):
#     """
#     扩展的GRPO Trainer, 支持在evaluation时计算答案相似度和Spearman相关性
#     """
    
#     def __init__(self, *args, **kwargs):
#         # 提取相似度评估配置
#         self.csts_eval_config = kwargs.pop('csts_eval_config', {})
        
#         # 在调用super().__init__之前初始化embedding模型
#         self._init_embedding_model()
        
#         super().__init__(*args, **kwargs)
        
#         # 存储整个evaluation过程的所有数据
#         self._reset_evaluation_data()
        
#         # 添加同步标志
#         self._eval_data_collection_complete = False
        
#     def _init_embedding_model(self):
#         """初始化embedding模型，确保只加载一次"""
#         if not self.csts_eval_config.get('enable_spearman_eval', False):
#             self.embedding_model = None
#             return

#         embedding_model = self.csts_eval_config.get('embedding_model')
        
#         from transformers.modeling_utils import set_zero3_state
#         with set_zero3_state():
#             self.embedding_model = SentenceTransformer(embedding_model)
#         logger.info(f"Initialized embedding model: {embedding_model}")
    
#     def _reset_evaluation_data(self):
#         """重置evaluation数据收集器"""
#         self._local_similarities = []
#         self._local_ground_truths = []
#         self._local_detailed_logs = []
#         self._eval_data_collection_complete = False
        
#         # 添加调试计数器
#         self._batch_count = 0
#         self._sample_count = 0
        
#     def _get_eval_sampler(self, eval_dataset):
#         """重写evaluation sampler，在evaluation时只生成1个completion"""
#         return RepeatSampler(
#             data_source=eval_dataset,
#             mini_repeat_count=1,  # evaluation时只生成1个completion
#             seed=self.args.seed,
#         )
        
#     def extract_answers(self, completion: str) -> Tuple[str, str]:
#         """从completion中提取两个答案"""
#         pattern1 = self.csts_eval_config.get('answer_pattern_1', r'<answer1>(.*?)</answer1>')
#         pattern2 = self.csts_eval_config.get('answer_pattern_2', r'<answer2>(.*?)</answer2>')
        
#         match1 = re.search(pattern1, completion, re.IGNORECASE)
#         match2 = re.search(pattern2, completion, re.IGNORECASE)
        
#         answer1 = match1.group(1).strip() if match1 else ""
#         answer2 = match2.group(1).strip() if match2 else ""
        
#         return answer1, answer2
    
#     def calculate_cosine_similarity(self, answer1: str, answer2: str) -> float:
#         """计算两个答案的余弦相似度"""
#         if not answer1 or not answer2:
#             return 0.0
        
#         embeddings = self.embedding_model.encode([answer1, answer2])
#         cos_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]).unsqueeze(0),
#             torch.tensor(embeddings[1]).unsqueeze(0)
#         ).item()
#         return cos_sim
        
#     def calculate_cosine_similarities_batch(self, answer_pairs: List[Tuple[str, str]]) -> List[float]:
#         """批量计算多个答案对的余弦相似度"""
#         if not answer_pairs:
#             return [0.0] * len(answer_pairs)
        
#         # 分离答案对
#         answers1 = []
#         answers2 = []
#         valid_indices = []
        
#         for i, (answer1, answer2) in enumerate(answer_pairs):
#             if answer1 and answer2:
#                 answers1.append(answer1)
#                 answers2.append(answer2)
#                 valid_indices.append(i)
        
#         if not answers1:
#             return [0.0] * len(answer_pairs)
        
#         # 批量编码
#         embeddings1 = self.embedding_model.encode(answers1, show_progress_bar=False)
#         embeddings2 = self.embedding_model.encode(answers2, show_progress_bar=False)
        
#         # 批量计算相似度
#         similarities = self.embedding_model.similarity_pairwise(embeddings1, embeddings2)
#         similarity_scores = similarities.tolist()
        
#         # 将结果映射回原始顺序
#         result = [0.0] * len(answer_pairs)
#         for i, valid_idx in enumerate(valid_indices):
#             result[valid_idx] = similarity_scores[i]
        
#         return result
    
#     # def _generate_and_score_completions(self, inputs):
#     #     """重写生成和评分方法，添加相似度评估"""
        
#     #     # 调用父类方法生成completions
#     #     result = super()._generate_and_score_completions(inputs)
        
#     #     # 在evaluation模式下收集相似度数据
#     #     mode = "train" if self.model.training else "eval"
#     #     if mode == "eval" and self.csts_eval_config.get('enable_spearman_eval', False):
#     #         self._collect_eval_data_local(inputs)
        
#     #     return result
    
#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
#         inputs = self._prepare_inputs(inputs)
#         if  self.csts_eval_config.get('enable_spearman_eval', False):
#             self._collect_eval_data_local(inputs)
#         with torch.no_grad():
#             with self.compute_loss_context_manager():
#                 loss = self.compute_loss(model, inputs)
#             loss = loss.mean().detach()
#         return loss, None, None
    
#     def _collect_eval_data_local(self, inputs):
#         """收集当前进程的评估数据（先存储答案对，后续批量计算）"""
#         try:
#             self._batch_count += 1
#             batch_size = len(inputs)
#             self._sample_count += batch_size
            
#             # if self.accelerator.is_main_process:
#             logger.info(f"Process {self.accelerator.process_index}: "
#                         f"Collecting batch {self._batch_count}, samples: {batch_size}, "
#                         f"total samples: {self._sample_count}")
#             # 从inputs中获取completions
#             completions = [inp['messages'][-1]['content'] for inp in inputs]
            
#             batch_answer_pairs = []
#             batch_ground_truths = []
#             batch_detailed_logs = []
            
#             for completion, inp in zip(completions, inputs):
#                 # 提取两个答案
#                 answer1, answer2 = self.extract_answers(completion)
#                 batch_answer_pairs.append((answer1, answer2))
                
#                 # 获取ground truth分数
#                 ground_truth_label = self.csts_eval_config.get('ground_truth_label', 'label')
#                 gt_score = inp.get(ground_truth_label, 0.0)
#                 if isinstance(gt_score, torch.Tensor):
#                     gt_score = gt_score.item()
#                 batch_ground_truths.append(gt_score)
                
#                 # 准备详细信息（相似度稍后计算）
#                 batch_detailed_logs.append({
#                     "completion": completion,
#                     "answer1": answer1,
#                     "answer2": answer2,
#                     "ground_truth_label": gt_score,
#                     "process_rank": self.accelerator.process_index,
#                 })
            
#             # 批量计算相似度
#             batch_similarities = self.calculate_cosine_similarities_batch(batch_answer_pairs)
            
#             # 更新详细日志中的预测相似度
#             for i, similarity in enumerate(batch_similarities):
#                 batch_detailed_logs[i]["predicted_similarity"] = similarity
            
#             # 存储到本地数据
#             self._local_similarities.extend(batch_similarities)
#             self._local_ground_truths.extend(batch_ground_truths)
            
#             if self.csts_eval_config.get('log_eval_details', True):
#                 self._local_detailed_logs.extend(batch_detailed_logs)
                    
#         except Exception as e:
#             logger.error(f"Process {self.accelerator.process_index}: "
#                         f"Error in batch {self._batch_count}: {e}")
#             if self.accelerator.is_main_process:
#                 logger.warning(f"Error during evaluation data collection: {e}")

#     def _gather_all_eval_data(self) -> Tuple[List[float], List[float], List[Dict]]:
#         """从所有进程收集evaluation数据，避免死锁"""
#         try:
#             # 使用更安全的同步方式
#             self.accelerator.wait_for_everyone()  # 确保所有进程到达这里
            
#             if self.accelerator.is_main_process:
#                 logger.info("Starting data gathering...")
            
#             # 方法1：只使用gather_object，避免tensor gather的复杂性
#             try:
#                 from accelerate.utils import gather_object
                
#                 # 所有进程都必须调用gather_object
#                 all_similarities_lists = gather_object(self._local_similarities)
#                 all_ground_truths_lists = gather_object(self._local_ground_truths)
#                 all_detailed_logs_lists = gather_object(self._local_detailed_logs)
                
#                 # 只在主进程处理结果
#                 if self.accelerator.is_main_process:
#                     # 扁平化数据
#                     all_similarities = []
#                     all_ground_truths = []
#                     all_detailed_logs = []
                    
#                     for similarities in all_similarities_lists:
#                         if isinstance(similarities, list):
#                             all_similarities.extend(similarities)
                    
#                     for ground_truths in all_ground_truths_lists:
#                         if isinstance(ground_truths, list):
#                             all_ground_truths.extend(ground_truths)
                    
#                     for logs in all_detailed_logs_lists:
#                         if isinstance(logs, list):
#                             all_detailed_logs.extend(logs)
                    
#                     logger.info(f"Gathered data from all processes: "
#                             f"{len(all_similarities)} similarities, "
#                             f"{len(all_ground_truths)} ground truths")
                    
#                     return all_similarities, all_ground_truths, all_detailed_logs
#                 else:
#                     # 非主进程返回空数据
#                     return [], [], []
                    
#             except Exception as e:
#                 if self.accelerator.is_main_process:
#                     logger.error(f"gather_object failed: {e}")
#                 # 备选：只使用本地数据
#                 return self._local_similarities, self._local_ground_truths, self._local_detailed_logs
        
#         except Exception as e:
#             if self.accelerator.is_main_process:
#                 logger.error(f"Data gathering failed: {e}")
#             return self._local_similarities, self._local_ground_truths, self._local_detailed_logs
    
#     def _compute_eval_metrics(self, all_pred_similarities: List[float], all_ground_truths: List[float]) -> Dict[str, float]:
#         """计算最终的相似度metrics"""
#         if not all_pred_similarities or not all_ground_truths:
#             return {}
        
#         try:
#             spearman_corr, p_value = spearmanr(all_pred_similarities, all_ground_truths)
#             spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
#         except:
#             spearman_corr = 0.0

#         try:
#             pearson_corr, _ = pearsonr(all_pred_similarities, all_ground_truths)
#             pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
#         except:
#             pearson_corr = 0.0
        
#         # 计算MAE和MSE
#         mae = np.mean(np.abs(np.array(all_pred_similarities) - np.array(all_ground_truths)))
#         mse = np.mean((np.array(all_pred_similarities) - np.array(all_ground_truths)) ** 2)
        
#         return {
#             "spearman_correlation": spearman_corr,
#             "pearson_correlation": pearson_corr,
#             "mae": mae,
#             "mse": mse,
#             "total_samples": len(all_pred_similarities),
#         }
    
#     def evaluation_loop(self, dataloader, *args, **kwargs):
#         """重写evaluation方法，确保数据收集完整性"""
        
#         # 添加调试信息
#         print(f"DEBUG: evaluation_loop called")
#         print(f"DEBUG: enable_spearman_eval = {self.csts_eval_config.get('enable_spearman_eval', False)}")
#         print(f"DEBUG: dataloader length = {len(dataloader)}")
        
#         # 重置evaluation数据收集器
#         self._reset_evaluation_data()
        
#         if self.accelerator.is_main_process:
#             logger.info(f"Starting evaluation with {len(dataloader)} batches")
        
#         # 调用父类evaluation方法
#         output = super().evaluation_loop(dataloader, *args, **kwargs)
        
#         # 添加调试信息
#         print(f"DEBUG: After super().evaluation_loop, metrics keys: {list(output.metrics.keys())}")
        
#         # 标记数据收集完成
#         self._eval_data_collection_complete = True
        
#         # 在evaluation完成后，计算基于所有样本的相似度metrics
#         enable_spearman = self.csts_eval_config.get('enable_spearman_eval', False)
#         print(f"DEBUG: About to check enable_spearman_eval: {enable_spearman}")
        
#         if enable_spearman:
#             print("DEBUG: Entering CSTS evaluation branch")
            
#             # 确保所有进程都完成了数据收集
#             self.accelerator.wait_for_everyone()
            
#             if self.accelerator.is_main_process:
#                 logger.info("All processes completed data collection, starting aggregation...")
            
#             # 所有进程都必须参与数据聚合
#             all_pred_similarities, all_ground_truths, all_detailed_logs = self._gather_all_eval_data()
            
#             print(f"DEBUG: Gathered data - similarities: {len(all_pred_similarities)}, ground_truths: {len(all_ground_truths)}")
            
#             # 只有主进程计算指标和记录日志
#             if self.accelerator.is_main_process:
#                 if len(all_pred_similarities) > 0:
#                     print("DEBUG: Computing CSTS metrics")
#                     csts_eval_metrics = self._compute_eval_metrics(all_pred_similarities, all_ground_truths)
#                     print(f"DEBUG: Computed metrics: {csts_eval_metrics}")
                    
#                     # 添加到返回的metrics中
#                     metric_key_prefix = kwargs.get('metric_key_prefix', 'eval')
#                     print(f"DEBUG: metric_key_prefix = {metric_key_prefix}")
                    
#                     for key, value in csts_eval_metrics.items():
#                         metric_key = f"{metric_key_prefix}_{key}"
#                         output.metrics[metric_key] = value
#                         print(f"DEBUG: Added metric {metric_key} = {value}")
                    
#                     print(f"DEBUG: Final metrics keys: {list(output.metrics.keys())}")
                    
#                     # 打印详细结果
#                     logger.info(f"\n=== Final Similarity Evaluation Results ===")
#                     logger.info(f"Total samples: {csts_eval_metrics.get('total_samples', 0)}")
#                     for key, value in csts_eval_metrics.items():
#                         if isinstance(value, float) and 'total_' not in key:
#                             logger.info(f"-- {key}: {value:.6f}")
#                     logger.info("=" * 50)
                    
#                     # 记录到wandb
#                     self._log_eval_results_to_wandb(csts_eval_metrics, all_detailed_logs)
#                 else:
#                     print("DEBUG: No evaluation data collected!")
#                     logger.warning("No evaluation data collected across all processes!")
            
#             # 最终同步
#             self.accelerator.wait_for_everyone()
#         else:
#             print("DEBUG: CSTS evaluation is disabled")
        
#         print(f"DEBUG: Final output.metrics keys: {list(output.metrics.keys())}")
#         return output

#     def _log_eval_results_to_wandb(self, eval_metrics: Dict[str, float], all_detailed_logs: List[Dict]):
#         """记录评估结果到wandb"""
#         if (self.args.report_to and "wandb" in self.args.report_to and 
#             all_detailed_logs and self.csts_eval_config.get('log_eval_details', True)):
            
#             try:
#                 import pandas as pd
#                 import wandb
                
#                 if wandb.run is not None:
#                     # 记录详细的样本级别数据
#                     similarity_df = pd.DataFrame(all_detailed_logs)
#                     wandb.log({
#                         "csts_evaluation_detailed": wandb.Table(dataframe=similarity_df),
#                         "step": self.state.global_step
#                     })
                    
#                     # 记录汇总统计
#                     wandb.log(eval_metrics, step=self.state.global_step)
                    
#             except Exception as e:
#                 logger.info(f"Error logging to wandb: {e}")



# import numpy as np
# import torch
# import re
# from trl.trainer.grpo_trainer import RepeatSampler
# from collections import defaultdict
# from typing import Any, Dict, List, Optional, Union, Tuple
# from scipy.stats import spearmanr, pearsonr
# from sentence_transformers import SentenceTransformer
# from .grpo_trainer import GRPOTrainer
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


# class CSTSGRPOTrainer(GRPOTrainer):
#     """
#     扩展的GRPO Trainer, 支持在evaluation时计算答案相似度和Spearman相关性
#     """
    
#     def __init__(self, *args, **kwargs):
#         # 提取相似度评估配置
#         self.csts_eval_config = kwargs.pop('csts_eval_config', {})
        
#         # 在调用super().__init__之前初始化embedding模型
#         self._init_embedding_model()
        
#         super().__init__(*args, **kwargs)
        
#         # 存储相似度预测和真实标签，用于prediction_step返回
#         self._current_batch_similarities = None
#         self._current_batch_ground_truths = None
        
#     def _init_embedding_model(self):
#         """初始化embedding模型，确保只加载一次"""
#         if not self.csts_eval_config.get('enable_spearman_eval', False):
#             self.embedding_model = None
#             return

#         embedding_model = self.csts_eval_config.get('embedding_model')
        
#         try:
#             from transformers.modeling_utils import set_zero3_state
#             with set_zero3_state():
#                 self.embedding_model = SentenceTransformer(embedding_model)
#         except:
#             # 如果set_zero3_state不可用，直接初始化
#             self.embedding_model = SentenceTransformer(embedding_model)
#         logger.info(f"Initialized embedding model: {embedding_model}")
        
#     def _get_eval_sampler(self, eval_dataset):
#         """重写evaluation sampler，在evaluation时只生成1个completion"""
#         return RepeatSampler(
#             data_source=eval_dataset,
#             mini_repeat_count=1,  # evaluation时只生成1个completion
#             seed=self.args.seed,
#         )
        
#     def extract_answers(self, completion: str) -> Tuple[str, str]:
#         """从completion中提取两个答案"""
#         pattern1 = self.csts_eval_config.get('answer_pattern_1', r'<answer1>(.*?)</answer1>')
#         pattern2 = self.csts_eval_config.get('answer_pattern_2', r'<answer2>(.*?)</answer2>')
        
#         match1 = re.search(pattern1, completion, re.IGNORECASE | re.DOTALL)
#         match2 = re.search(pattern2, completion, re.IGNORECASE | re.DOTALL)
        
#         answer1 = match1.group(1).strip() if match1 else ""
#         answer2 = match2.group(1).strip() if match2 else ""
        
#         return answer1, answer2
    
#     def calculate_cosine_similarities_batch(self, answer_pairs: List[Tuple[str, str]]) -> List[float]:
#         """批量计算多个答案对的余弦相似度"""
#         if not answer_pairs or self.embedding_model is None:
#             return [0.0] * len(answer_pairs)
        
#         # 分离答案对
#         answers1 = []
#         answers2 = []
#         valid_indices = []
        
#         for i, (answer1, answer2) in enumerate(answer_pairs):
#             if answer1.strip() and answer2.strip():
#                 answers1.append(answer1.strip())
#                 answers2.append(answer2.strip())
#                 valid_indices.append(i)
        
#         if not answers1:
#             return [0.0] * len(answer_pairs)
        
#         try:
#             # 批量编码
#             embeddings1 = self.embedding_model.encode(answers1, show_progress_bar=False)
#             embeddings2 = self.embedding_model.encode(answers2, show_progress_bar=False)
            
#             # 批量计算相似度
#             similarities = self.embedding_model.similarity_pairwise(embeddings1, embeddings2)
#             similarity_scores = similarities.tolist()
            
#             # 将结果映射回原始顺序
#             result = [0.0] * len(answer_pairs)
#             for i, valid_idx in enumerate(valid_indices):
#                 result[valid_idx] = similarity_scores[i]
            
#             return result
#         except Exception as e:
#             logger.error(f"Error in batch similarity calculation: {e}")
#             return [0.0] * len(answer_pairs)

#     def _prepare_inputs(self, inputs):
#         """重写_prepare_inputs，在evaluation时计算相似度并保存到inputs中"""
        
#         # 调用父类方法获取标准的tensor化输入
#         prepared_inputs = super()._prepare_inputs(inputs)
        
#         # 如果是evaluation模式且启用了CSTS评估，计算相似度
#         if (not self.model.training and 
#             self.csts_eval_config.get('enable_spearman_eval', False)):
            
#             try:
#                 similarities, ground_truths = self._compute_similarities_from_inputs(prepared_inputs)
                
#                 # 确保tensor在正确的设备上
#                 # device = next(iter(prepared_inputs[0].values())).device if prepared_inputs else torch.device('cpu')
                
#                 if len(similarities) > 0:
#                     similarities_tensor = torch.tensor(similarities, dtype=torch.float32)
#                     ground_truths_tensor = torch.tensor(ground_truths, dtype=torch.float32)
                    
#                     # 保存到当前batch中，供prediction_step使用
#                     self._current_batch_similarities = similarities_tensor
#                     self._current_batch_ground_truths = ground_truths_tensor
#                 else:
#                     self._current_batch_similarities = torch.tensor([], dtype=torch.float32)
#                     self._current_batch_ground_truths = torch.tensor([], dtype=torch.float32)
                    
#             except Exception as e:
#                 logger.error(f"Error computing similarities in _prepare_inputs: {e}")
#                 # device = next(iter(prepared_inputs[0].values())).device if prepared_inputs else torch.device('cpu')
#                 self._current_batch_similarities = torch.tensor([], dtype=torch.float32)
#                 self._current_batch_ground_truths = torch.tensor([], dtype=torch.float32)
        
#         return prepared_inputs

#     def _compute_similarities_from_inputs(self, inputs):
#         """从inputs中计算相似度和真实标签"""
#         try:
#             similarities = []
#             ground_truths = []
            
#             for inp in inputs:
#                 # logger.info(f"Processing input: {inp['messages'][-1]['content'][:20]}...")  # 日志前50个字符
#                 # 提取completion文本（不变）
#                 completion = ""
#                 if 'messages' in inp and isinstance(inp['messages'], list):
#                     for msg in reversed(inp['messages']):
#                         if isinstance(msg, dict) and msg.get('role') == 'assistant':
#                             completion = msg.get('content', '')
#                             break
#                 elif 'completion' in inp:
#                     completion = inp['completion']
#                 # 提取答案对
#                 answer1, answer2 = self.extract_answers(completion)
                
#                 # 添加日志诊断
#                 logger.info(f"Process {self.accelerator.process_index}: Extracted answer1='{answer1[:50]}...', answer2='{answer2[:50]}...' from completion='{completion[:50]}...'")
                
#                 if not answer1.strip() or not answer2.strip():
#                     logger.info(f"Process {self.accelerator.process_index}: One or both answers are empty, skipping similarity calculation.")
#                     similarities.append(0.0)
#                 else:
#                     try:
#                         embeddings1 = self.embedding_model.encode([answer1.strip()], show_progress_bar=False)
#                         embeddings2 = self.embedding_model.encode([answer2.strip()], show_progress_bar=False)
                        
#                         # 新增：检查 norm，避免 division by zero
#                         norm1 = np.linalg.norm(embeddings1)
#                         norm2 = np.linalg.norm(embeddings2)
#                         if norm1 == 0 or norm2 == 0:
#                             similarity = 0.0
#                         else:
#                             similarity = self.embedding_model.similarity_pairwise(embeddings1, embeddings2).item()
#                             if np.isnan(similarity):  # 额外安全检查
#                                 similarity = 0.0
                        
#                         similarities.append(similarity)
#                     except Exception as e:
#                         logger.error(f"Error computing similarity for single pair: {e}")
#                         similarities.append(0.0)
                
#                 # 提取ground truth（不变）
#                 ground_truth_label = self.csts_eval_config.get('ground_truth_label', 'label')
#                 gt_score = inp.get(ground_truth_label, 0.0)
#                 if isinstance(gt_score, torch.Tensor):
#                     gt_score = gt_score.item()
#                 ground_truths.append(float(gt_score))
            
#             return similarities, ground_truths
            
#         except Exception as e:
#             logger.error(f"Error in _compute_similarities_from_inputs: {e}")
#             return [], []

#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
#         """修改后的prediction_step，返回相似度预测和真实标签"""
        
#         # 调用_prepare_inputs，这会计算相似度并保存到self._current_batch_*
#         inputs = self._prepare_inputs(inputs)
        
#         with torch.no_grad():
#             with self.compute_loss_context_manager():
#                 loss = self.compute_loss(model, inputs)
#             loss = loss.mean().detach()
        
#         # 检查是否有CSTS相似度信息
#         if (self.csts_eval_config.get('enable_spearman_eval', False) and 
#             self._current_batch_similarities is not None and 
#             self._current_batch_ground_truths is not None):
            
#             similarities = self._current_batch_similarities
#             ground_truths = self._current_batch_ground_truths
            
#             # 确保在正确的设备上
#             if similarities.device != loss.device:
#                 similarities = similarities.to(loss.device)
#             if ground_truths.device != loss.device:
#                 ground_truths = ground_truths.to(loss.device)
            
#             # 清空当前batch的数据
#             self._current_batch_similarities = None
#             self._current_batch_ground_truths = None
            
#             logger.info(f"Returning CSTS predictions: similarities shape={similarities.shape}, ground_truths shape={ground_truths.shape}")
            
#             return loss, similarities, ground_truths
#         else:
#             # 如果没有CSTS信息，返回原来的格式
#             return loss, None, None








import numpy as np
import torch
import re
from trl.trainer.grpo_trainer import RepeatSampler
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from .grpo_trainer import GRPOTrainer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CSTSGRPOTrainer(GRPOTrainer):
    """
    扩展的GRPO Trainer, 支持在evaluation时计算答案相似度和Spearman相关性
    """
    
    def __init__(self, *args, **kwargs):
        # 提取相似度评估配置
        self.csts_eval_config = kwargs.pop('csts_eval_config', {})
        
        # 在调用super().__init__之前初始化embedding模型
        self._init_embedding_model()
        
        super().__init__(*args, **kwargs)
        
        # 存储相似度预测和真实标签，用于prediction_step返回
        self._current_batch_similarities = None
        self._current_batch_ground_truths = None
        
    def _init_embedding_model(self):
        """初始化embedding模型，确保只加载一次"""
        if not self.csts_eval_config.get('enable_spearman_eval', False):
            self.embedding_model = None
            return

        embedding_model = self.csts_eval_config.get('embedding_model')
        
        try:
            from transformers.modeling_utils import set_zero3_state
            with set_zero3_state():
                self.embedding_model = SentenceTransformer(embedding_model)
        except:
            # 如果set_zero3_state不可用，直接初始化
            self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized embedding model: {embedding_model}")
        
    # def get_eval_dataloader(self, eval_dataset=None):
    #     """重写evaluation dataloader，确保不丢弃最后一个batch"""
    #     if eval_dataset is None and self.eval_dataset is None:
    #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
    #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
    #     if self.csts_eval_config.get('enable_spearman_eval', False):
    #         from torch.utils.data import DataLoader
            
    #         eval_dataloader = DataLoader(
    #             eval_dataset,
    #             batch_size=self.args.eval_batch_size,
    #             sampler=self._get_eval_sampler(eval_dataset),
    #             collate_fn=self.data_collator,
    #             drop_last=False,  # 关键修改：不丢弃最后一个batch
    #             num_workers=self.args.dataloader_num_workers,
    #             pin_memory=self.args.dataloader_pin_memory,
    #         )
            
    #         logger.info(f"CSTS DataLoader: {len(eval_dataset)}样本, {len(eval_dataloader)}batches, drop_last=False")
    #         return eval_dataloader
    #     else:
    #         return super().get_eval_dataloader(eval_dataset)
        
    def _get_eval_sampler(self, eval_dataset):
        """重写evaluation sampler，在evaluation时只生成1个completion"""
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,  # evaluation时只生成1个completion
            seed=self.args.seed,
        )
        
    def extract_answers(self, completion: str) -> Tuple[str, str]:
        """从completion中提取两个答案"""
        pattern1 = self.csts_eval_config.get('answer_pattern_1', r'<answer1>(.*?)</answer1>')
        pattern2 = self.csts_eval_config.get('answer_pattern_2', r'<answer2>(.*?)</answer2>')
        
        match1 = re.search(pattern1, completion, re.IGNORECASE | re.DOTALL)
        match2 = re.search(pattern2, completion, re.IGNORECASE | re.DOTALL)
        
        answer1 = match1.group(1).strip() if match1 else ""
        answer2 = match2.group(1).strip() if match2 else ""
        
        return answer1, answer2
    
    def calculate_cosine_similarities_batch(self, answer_pairs: List[Tuple[str, str]]) -> List[float]:
        """批量计算多个答案对的余弦相似度"""
        if not answer_pairs or self.embedding_model is None:
            return [0.0] * len(answer_pairs)
        
        # 分离答案对
        answers1 = []
        answers2 = []
        valid_indices = []
        
        for i, (answer1, answer2) in enumerate(answer_pairs):
            if answer1.strip() and answer2.strip():
                answers1.append(answer1.strip())
                answers2.append(answer2.strip())
                valid_indices.append(i)
        
        if not answers1:
            return [0.0] * len(answer_pairs)
        
        try:
            # 批量编码
            embeddings1 = self.embedding_model.encode(answers1, show_progress_bar=False)
            embeddings2 = self.embedding_model.encode(answers2, show_progress_bar=False)
            
            # 批量计算相似度
            similarities = self.embedding_model.similarity_pairwise(embeddings1, embeddings2)
            similarity_scores = similarities.tolist()
            
            # 将结果映射回原始顺序
            result = [0.0] * len(answer_pairs)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = similarity_scores[i]
            
            return result
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            return [0.0] * len(answer_pairs)

    def _generate_and_score_completions(self, inputs):
        """重写生成和评分方法，在evaluation时添加相似度计算"""
        
        # 调用父类方法获取标准结果
        result = super()._generate_and_score_completions(inputs)
        
        # 在evaluation模式下，计算相似度数据
        mode = 'train' if self.model.training else 'eval'
        if (mode == 'eval' and 
            self.csts_eval_config.get('enable_spearman_eval', False)):
            
            try:
                # 此时inputs已经包含了生成的completions
                similarities, ground_truths = self._compute_similarities_from_inputs_with_completions(inputs)
                
                # 保存到当前batch中，供prediction_step使用
                device = torch.device('cpu')  # 先用CPU，后面会移动到正确设备
                if len(similarities) > 0:
                    self._current_batch_similarities = torch.tensor(similarities, dtype=torch.float32, device=device)
                    self._current_batch_ground_truths = torch.tensor(ground_truths, dtype=torch.float32, device=device)
                else:
                    self._current_batch_similarities = torch.tensor([], dtype=torch.float32, device=device)
                    self._current_batch_ground_truths = torch.tensor([], dtype=torch.float32, device=device)
                    
                logger.info(f"Process {self.accelerator.process_index}: "
                          f"Computed {len(similarities)} similarities for evaluation")
                    
            except Exception as e:
                logger.error(f"Error computing similarities in _generate_and_score_completions: {e}")
                device = torch.device('cpu')
                self._current_batch_similarities = torch.tensor([], dtype=torch.float32, device=device)
                self._current_batch_ground_truths = torch.tensor([], dtype=torch.float32, device=device)
        
        return result

    def _compute_similarities_from_inputs_with_completions(self, inputs):
        """从包含completions的inputs中计算相似度和真实标签"""
        try:
            similarities = []
            ground_truths = []
            
            logger.info(f"Process {self.accelerator.process_index}: "
                       f"Processing {len(inputs)} inputs with completions")
            
            for i, inp in enumerate(inputs):
                # 提取completion文本（此时messages已经包含生成的completion）
                completion = ""
                if 'messages' in inp and isinstance(inp['messages'], list):
                    # 获取最后一条assistant消息
                    for msg in reversed(inp['messages']):
                        if isinstance(msg, dict) and msg.get('role') == 'assistant':
                            completion = msg.get('content', '')
                            break
                
                if not completion:
                    logger.warning(f"Process {self.accelerator.process_index}: "
                                 f"No completion found in input {i}")
                    similarities.append(0.0)
                    ground_truths.append(0.0)
                    continue
                
                # 提取答案对并计算相似度
                answer1, answer2 = self.extract_answers(completion)
                
                # 添加调试信息（只对前几个样本）
                if i < 3:
                    logger.info(f"Process {self.accelerator.process_index}: "
                               f"Input {i}: answer1='{answer1[:30]}...', "
                               f"answer2='{answer2[:30]}...'")
                
                if answer1.strip() and answer2.strip() and self.embedding_model is not None:
                    try:
                        embeddings1 = self.embedding_model.encode([answer1.strip()], show_progress_bar=False)
                        embeddings2 = self.embedding_model.encode([answer2.strip()], show_progress_bar=False)
                        
                        # 检查 norm，避免 division by zero
                        norm1 = np.linalg.norm(embeddings1)
                        norm2 = np.linalg.norm(embeddings2)
                        if norm1 == 0 or norm2 == 0:
                            similarity = 0.0
                        else:
                            similarity = self.embedding_model.similarity_pairwise(embeddings1, embeddings2)[0].item()
                            if np.isnan(similarity):
                                similarity = 0.0
                        
                        similarities.append(similarity)
                    except Exception as e:
                        logger.error(f"Error computing similarity for input {i}: {e}")
                        similarities.append(0.0)
                else:
                    similarities.append(0.0)
                
                # 提取ground truth
                ground_truth_label = self.csts_eval_config.get('ground_truth_label', 'label')
                gt_score = inp.get(ground_truth_label, 0.0)
                if isinstance(gt_score, torch.Tensor):
                    gt_score = gt_score.item()
                ground_truths.append(float(gt_score))
            
            logger.info(f"Process {self.accelerator.process_index}: "
                       f"Successfully computed {len(similarities)} similarities, "
                       f"{len(ground_truths)} ground truths")
            
            return similarities, ground_truths
            
        except Exception as e:
            logger.error(f"Process {self.accelerator.process_index}: "
                        f"Error in _compute_similarities_from_inputs_with_completions: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], []

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        """修改后的prediction_step，返回相似度预测和真实标签"""
        
        # 调用_prepare_inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        
        # 检查是否有CSTS相似度信息
        if (self.csts_eval_config.get('enable_spearman_eval', False) and 
            self._current_batch_similarities is not None and 
            self._current_batch_ground_truths is not None):
            
            similarities = self._current_batch_similarities
            ground_truths = self._current_batch_ground_truths
            
            # 确保在正确的设备上
            if similarities.device != loss.device:
                similarities = similarities.to(loss.device)
            if ground_truths.device != loss.device:
                ground_truths = ground_truths.to(loss.device)
            
            # 添加调试信息
            logger.info(f"Process {self.accelerator.process_index}: "
                       f"Returning CSTS predictions: similarities shape={similarities.shape}, "
                       f"ground_truths shape={ground_truths.shape}")
            
            # 清空当前batch的数据
            self._current_batch_similarities = None
            self._current_batch_ground_truths = None
            
            return loss, similarities, ground_truths
        else:
            # 如果没有CSTS信息，返回原来的格式
            return loss, None, None