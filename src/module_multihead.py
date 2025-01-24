import copy
import json
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BeamSearchScorer,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput


class Solomon(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def init_prompt(self, num_prompts, prompt_length, num_heads, device, user_to_embeddings):
        self.model_device = device
        self.user_to_embeddings = user_to_embeddings
        self.num_heads = num_heads
        self.emsize = self.config.hidden_size  # from self.config
        self.prompt_length = prompt_length
        self.num_prompts = num_prompts
        self.num_tasks = 3
        self.att_dim = self.emsize  # dimension for attention

        self.whole_word_embeddings = nn.Embedding(self.config.n_positions, self.emsize)

        # Q, K, V projection layers
        self.query_layer = nn.Linear(512, self.att_dim)
        self.key_layer = nn.Linear(512, self.att_dim)
        self.value_layer = nn.Linear(512, self.att_dim)

        # Map attended context to prompt embeddings
        self.prompt_projector = nn.Linear(self.att_dim, self.prompt_length * self.emsize)

        # Task embeddings
        self.task_embeddings = nn.Parameter(
            torch.empty(self.num_tasks, 3, self.emsize).uniform_(-0.1, 0.1)
        )
        self.precompute_cosine_similarities()

    def precompute_cosine_similarities(self):
        """
        Precompute cosine similarity for all users and store them in a dictionary.
        """
        # Extract all user embeddings
        user_ids = list(self.user_to_embeddings.keys())
        user_embs = torch.tensor(
            [self.user_to_embeddings[uid] for uid in user_ids], dtype=torch.float, device=self.model_device
        )  # [N, d]

        # Normalize embeddings
        normalized_user_embs = F.normalize(user_embs, dim=1)  # [N, d]

        # Compute cosine similarity matrix
        cos_sim_matrix = torch.matmul(normalized_user_embs, normalized_user_embs.T)  # [N, N]

        # Store the precomputed similarities
        self.precomputed_similarities = {}
        for i, uid in tqdm(enumerate(user_ids)):
            self.precomputed_similarities[uid] = {
                "user_ids": user_ids,
                "cos_sim": cos_sim_matrix[i].cpu().numpy(),  # Store as numpy array for easier indexing
            }
        
        self.user_ids = user_ids

    def get_top_k_similar_users(self, target_user_id, k):
        """
        Retrieve the top-K similar users for a given user using precomputed similarities.
        """
        if str(target_user_id) not in self.precomputed_similarities:
            raise ValueError(f"User ID {target_user_id} not found in precomputed similarities.")

        # Access precomputed similarities
        similarity_data = self.precomputed_similarities[str(target_user_id)]
        all_cos_sim = similarity_data["cos_sim"]  # [N]
        user_ids = similarity_data["user_ids"]

        # Find the top-k indices (exclude the target user itself)
        target_idx = self.user_ids.index(str(target_user_id))
        all_cos_sim[target_idx] = float('-inf')  # Exclude self-similarity
        top_indices = torch.topk(torch.tensor(all_cos_sim), k=k).indices

        # Retrieve the top-k user embeddings and IDs
        top_user_ids = [user_ids[idx] for idx in top_indices.tolist()]
        top_user_embs = torch.tensor(
            [self.user_to_embeddings[uid] for uid in top_user_ids], dtype=torch.float, device=self.model_device
        )

        # Retrieve the corresponding similarity values
        top_values = all_cos_sim[top_indices].tolist()

        return top_user_ids, top_user_embs, top_values


    def input_plus_whole_word(self, input_ids, whole_word_ids):
        text_emb = self.shared(input_ids)  # (batch_size, src_len, emsize)
        whole_word_emb = self.whole_word_embeddings(whole_word_ids)
        text_emb_plus = text_emb + whole_word_emb
        return text_emb_plus

    def append_prompt(
        self, task_id, user_ids, input_ids, whole_word_ids, attention_mask
    ):
        batch_size = input_ids.size(0)
        text_emb_plus = self.input_plus_whole_word(input_ids, whole_word_ids)

        # Get target users' embeddings
        # [batch_size, d]
        users_emb = torch.stack(
            [torch.tensor(self.user_to_embeddings[str(uid)], dtype=torch.float) for uid in user_ids]
        ).to(self.model_device)

        # For each user in the batch, get top-k similar users
        similar_emb_list = []
        for uid in user_ids:
            _, top_embs, _ = self.get_top_k_similar_users(uid, self.num_prompts)  # top_embs: [k, d]
            similar_emb_list.append(top_embs)

        # Stack all similar users embeddings into [batch_size, k, d]
        similar_users_emb = torch.stack(similar_emb_list, dim=0).to(self.model_device)

        # Multi-head attention setup
        head_dim = self.att_dim // self.num_heads
        assert self.att_dim % self.num_heads == 0, "att_dim must be divisible by num_heads"

        # Q: from target user embeddings, split into heads
        Q = self.query_layer(users_emb)  # [b, att_dim]
        Q = Q.view(batch_size, self.num_heads, head_dim).transpose(0, 1)  # [num_heads, b, head_dim]

        # K, V: from similar users' embeddings, split into heads
        K = self.key_layer(similar_users_emb)  # [b, k, att_dim]
        V = self.value_layer(similar_users_emb)  # [b, k, att_dim]

        K = K.view(batch_size, -1, self.num_heads, head_dim).permute(2, 0, 1, 3)  # [num_heads, b, k, head_dim]
        V = V.view(batch_size, -1, self.num_heads, head_dim).permute(2, 0, 1, 3)  # [num_heads, b, k, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-1, -2))  # [num_heads, b, 1, k]
        scores = scores / math.sqrt(head_dim)
        att_weights = torch.softmax(scores, dim=-1)  # [num_heads, b, 1, k]

        # Weighted sum of V
        att_context = torch.matmul(att_weights, V)  # [num_heads, b, 1, head_dim]
        att_context = att_context.squeeze(2).transpose(0, 1)  # [b, num_heads, head_dim]

        # Concatenate heads and project back to original dimension
        att_context = att_context.reshape(batch_size, -1)  # [b, att_dim]

        # project prompt to prompt_length
        projected_output = self.prompt_projector(att_context).reshape(batch_size, self.prompt_length, self.emsize)  # Shape: (batch_size, prompt_length * embedding_dim)

        # Task-specific prompts
        task_prompt = self.task_embeddings[task_id]

        # Concatenate prompts and text
        input_emb = torch.cat([task_prompt, projected_output, text_emb_plus], dim=1)

        # Adjust attention mask
        total_prompt_len = 3 + self.prompt_length  # task_prompt + attended_prompts
        prompt_pad = torch.ones((batch_size, total_prompt_len), dtype=attention_mask.dtype, device=self.model_device)
        input_mask = torch.cat([prompt_pad, attention_mask], dim=1)

        return input_emb, input_mask


    def forward(
        self,
        task_id=None,
        user_ids=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if encoder_outputs is None:
            input_emb, attention_mask = self.append_prompt(
                task_id, user_ids, input_ids, whole_word_ids, attention_mask
            )
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                # input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=input_emb,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return super().forward(
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def beam_search(
        self,
        task_id=None,
        user_ids=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        max_length=50,
        num_beams=20,
        num_beam_groups=1,
        early_stopping=True,
        min_length=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        num_return_sequences=20,
        bad_words_ids=None,
    ):
        # define decoder start token ids
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.ones(
            (num_beams * batch_size, 1), dtype=torch.int64
        ).to(self.model_device)
        decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        input_emb, attention_mask = self.append_prompt(
            task_id, user_ids, input_ids, whole_word_ids, attention_mask
        )
        model_kwargs = {
            "encoder_outputs": self.encoder(
                attention_mask=attention_mask.repeat_interleave(num_beams, dim=0),
                inputs_embeds=input_emb.repeat_interleave(num_beams, dim=0),
                return_dict=True,
            )
        }

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.model_device,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences,
            do_early_stopping=early_stopping,
        )

        criteria = StoppingCriteriaList()
        criteria.append(MaxLengthCriteria(max_length=max_length))

        # instantiate logits processors
        logits_processor = LogitsProcessorList()
        logits_processor.append(
            MinLengthLogitsProcessor(min_length, eos_token_id=self.config.eos_token_id)
        )
        if bad_words_ids is not None:
            logits_processor.append(
                NoBadWordsLogitsProcessor(
                    bad_words_ids, eos_token_id=self.config.eos_token_id
                )
            )
        self.generation_config.output_logits = None

        if num_beam_groups == 1:
            return super().beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs,
            )
        else:
            if diversity_penalty > 0.0:
                logits_processor.append(
                    HammingDiversityLogitsProcessor(
                        diversity_penalty,
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups,
                    )
                )
            if repetition_penalty != 1.0:
                logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(
                        penalty=repetition_penalty,
                    )
                )

            return super().group_beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs,
            )
