from typing import List, Optional
from pydantic import BaseModel


class CompressPromptRequest(BaseModel):
    context: Optional[List[str]] = None
    instruction: Optional[str] = ""
    question: Optional[str] = ""
    ratio: Optional[float] = 0.5
    target_token: Optional[float] = -1
    iterative_size: Optional[int] = 200
    force_context_ids: Optional[List[int]] = None
    force_context_number: Optional[int] = None
    use_sentence_level_filter: Optional[bool] = False
    use_context_level_filter: Optional[bool] = True
    use_token_level_filter: Optional[bool] = True
    keep_split: Optional[bool] = False
    keep_first_sentence: Optional[int] = 0
    keep_last_sentence: Optional[int] = 0
    keep_sentence_number: Optional[int] = 0
    high_priority_bonus: Optional[int] = 100
    context_budget: Optional[str] = "+100"
    token_budget_ratio: Optional[float] = 1.4
    condition_in_question: Optional[str] = "none"
    reorder_context: Optional[str] = "original"
    dynamic_context_compression_ratio: Optional[float] = 0.0
    condition_compare: Optional[bool] = False
    add_instruction: Optional[bool] = False
    rank_method: Optional[str] = "longllmlingua"
    concate_question: Optional[bool] = True
