import torch
import torch.nn as nn
from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.modeling_outputs import (
    BaseModelOutput, MaskedLMOutput, MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput,
)

from utils.layers import GAULayer, Norm

logger = logging.get_logger(__name__)


def initializer(tensor, num_hidden_layers=24, order=2, gain=1.0):
    """使用截断正态分布初始化
    """
    shape = tensor.shape
    if shape[0] > 10000 or shape[0] < 10:
        hidden_size = shape[1]
    else:
        hidden_size = shape[0]
    gain *= num_hidden_layers ** (-1. / order)
    std = 1.13684723 / hidden_size ** 0.5 * gain
    return nn.init.trunc_normal_(tensor, std=std)


class GAUConfig(PretrainedConfig):
    model_type = "gau"
    
    def __init__(
            self,
            vocab_size=12000,
            hidden_size=768,
            intermediate_size=1536,
            num_hidden_layers=24,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            attention_key_size=128,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            hidden_act="swish",
            classifier_dropout=0.1,
            use_bias=False,
            normalization="softmax_plus",
            attention_scale=True,
            embedding_size=None,
            scaling_factor="n",
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_key_size = attention_key_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.gradient_checkpointing = gradient_checkpointing
        self.classifier_dropout = classifier_dropout
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.intermediate_size = intermediate_size
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.scaling_factor = scaling_factor


class GAUPreTrainedModel(PreTrainedModel):
    config_class = GAUConfig
    base_model_prefix = "gau"
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            initializer(module.weight.data, self.config.num_hidden_layers, order=2, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            initializer(module.weight.data, self.config.num_hidden_layers, order=2, gain=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GAUEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
        
        self.norm = Norm(eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids=None, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GAUEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                GAULayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    attention_key_size=config.attention_key_size,
                    activation=config.hidden_act,
                    use_bias=config.use_bias,
                    normalization=config.normalization,
                    attention_scale=config.attention_scale,
                    attention_dropout=config.attention_probs_dropout_prob,
                    hidden_dropout=config.hidden_dropout_prob,
                    eps=config.layer_norm_eps,
                    scaling_factor=config.scaling_factor,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        sinusoidal_id = self.get_sinusoidal_id(
            config.max_position_embeddings, config.attention_key_size
        )
        self.register_buffer("sin_pos", sinusoidal_id.sin(), persistent=False)
        self.register_buffer("cos_pos", sinusoidal_id.cos(), persistent=False)
    
    def get_sinusoidal_id(self, max_length, output_dim):
        position_ids = torch.arange(0, max_length, dtype=torch.float32)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        sinusoidal_id = torch.einsum("n,d->nd", position_ids, indices)
        return sinusoidal_id[None, :, :]
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        seqlen = hidden_states.shape[1]
        sinusoidal_pos = self.sin_pos[:, :seqlen, :], self.cos_pos[:, :seqlen, :]
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GAUModel(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = GAUEmbeddings(config)
        self.encoder = GAUEncoder(config)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).type_as(
                self.embeddings.word_embeddings.weight
            )
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RoFormerV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
    
    def forward(self, hidden_states):
        return self.decoder(hidden_states)


class GAUOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RoFormerV2LMPredictionHead(config)
    
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class GAUClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
        self.config = config
    
    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GAUForMaskedLM(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.gau = GAUModel(config)
        self.cls = GAUOnlyMLMHead(config)
        
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.gau(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )
        
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )
        
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GAUForSequenceClassification(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gau = GAUModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = GAUClassificationHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.gau(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GAUForMultipleChoice(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.gau = GAUModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = input_ids.shape[1]
        input_ids = (
            input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.reshape(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.reshape(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        
        outputs = self.gau(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = self.sequence_summary(outputs[0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GAUForTokenClassification(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gau = GAUModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.gau(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GAUForQuestionAnswering(GAUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.gau = GAUModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.gau(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
