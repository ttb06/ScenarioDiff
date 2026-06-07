from transformers import BertConfig, BertModel, BertTokenizer


DESCRIPTIONS = {
    "Economy": ["International Trade Balance", "month"],
    "Energy": ["Gasoline Prices", "week"],
    "Security": ["Disaster and Emergency Grants", "month"],
    "SocialGood": ["Unemployment Rate", "month"],
    "Traffic": ["Travel Volume", "month"],
}


def get_desc(domain, lookback_len, pred_len):
    target, frequency = DESCRIPTIONS[domain]
    return (
        f"Below is historical reporting information over the past {lookback_len} "
        f"{frequency}s concerning the {target}. Based on these reports, predict "
        f"the potential trends and anomalies of the {target} for the next "
        f"{pred_len} {frequency}s."
    )


def get_llm(llm_model: str, llm_layers: int = 0):
    if llm_model != "bert":
        raise ValueError("Only the frozen BERT text encoder is supported.")

    model_name = "google-bert/bert-base-uncased"
    config = BertConfig.from_pretrained(model_name, local_files_only=False)
    if llm_layers:
        config.num_hidden_layers = llm_layers
    config.output_attentions = True
    config.output_hidden_states = True

    try:
        model = BertModel.from_pretrained(
            model_name,
            local_files_only=True,
            config=config,
        )
    except Exception:
        model = BertModel.from_pretrained(
            model_name,
            local_files_only=False,
            config=config,
        )

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=False)

    return model, tokenizer
