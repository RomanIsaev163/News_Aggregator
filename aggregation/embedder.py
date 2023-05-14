from transformers import AutoTokenizer, AutoModel
import torch

# Функции получения эмбэддингов
def mean_pooling(model_output, attention_mask, device):
    token_embeddings = model_output[0].to(device) #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(device)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
def embed_bert_cls(model_output, attention_mask, device):
    """Получение эмбеддингов из текста"""
    embeddings = model_output.last_hidden_state[:, 0, :].to(device)
    embeddings = torch.nn.functional.normalize(embeddings).to(device)
    return embeddings


class Encoder:
    def __init__(self, model_name, tokenizer='cointegrated/rubert-tiny2'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model_name)
        print(f'model_name: {model_name}')

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = self.model.device

    def encoding_sentence(self, text, embedding_func, max_length=100):
        """Получение эмбеддингов из текста"""

        encoded_input = self.tokenizer(text.replace('\n', ' '), padding=True, truncation=True, return_tensors='pt',
                                       max_length=max_length)
        # encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
            # model_output = self.model(**encoded_input)
        embeddings = embedding_func(model_output, encoded_input['attention_mask'], self.device)

        return embeddings[0].cpu().numpy()

    def encode_data(self, text_pool, embedding_func=embed_bert_cls, data_column='content', ):
        print("Encoding data...")
        """
            Вход:
                content - содержимое новости
                date - дата публикации
            Выход:
                embeddings_pool- list с эмбеддингами новостей
        """
        embeddings_pool = text_pool.apply(
            lambda x: self.encoding_sentence(x[data_column], embedding_func=embedding_func), axis=1
        )
        print("Encoding done!")
        print(f'Размер embedding: {embeddings_pool.iloc[0].shape}')
        return embeddings_pool.values