import torch
from transformers import AutoConfig, AutoTokenizer, RobertaModel

def create_temp_rel_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    special = ["[T1]", "[/T1]", "[T2]", "[/T2]", "[TIMEX3]", "[/TIMEX3]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    return tokenizer

class TemporalRelationsModel(torch.nn.Module):
    def __init__(self, num_labels, tokenizer, t1 = "[T1]", t2 = "[T2]"):
        super(TemporalRelationsModel, self).__init__()
        
        config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
        config.is_decoder = False
        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base", config=config)
        self.roberta.resize_token_embeddings(len(tokenizer))

        self.t1_id = tokenizer.convert_tokens_to_ids(t1)
        self.t2_id = tokenizer.convert_tokens_to_ids(t2)

        hidden_size = self.roberta.config.hidden_size
        hidden_dropout_prob = self.roberta.config.hidden_dropout_prob

        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size * 4, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        hidden = outputs.last_hidden_state
        t1_mask = (input_ids == self.t1_id)
        t2_mask = (input_ids == self.t2_id)

        h1 = hidden[t1_mask]
        h2 = hidden[t2_mask]

        pair = torch.cat(
            [h1, h2, h1 - h2, h1 * h2],
            dim=1
        )

        logits = self.classifier(pair)
        return logits