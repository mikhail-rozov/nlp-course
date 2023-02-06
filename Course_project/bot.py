import annoy
import logging
import pandas as pd
import py3langid as langid
import requests
import spacy
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, MessageHandler
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Токены для API
from data.tokens import coord_token, weather_token, bot_token

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

coords_url = 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/suggest/address'
coords_headers = {'Content-Type': 'application/json',
                  'Accept': 'application/json',
                  'Authorization': f'Token {coord_token}'}

weather_url = 'https://api.weather.yandex.ru/v2/informers/'
weather_headers = {'X-Yandex-API-Key': f'{weather_token}'}

# Перевод на русский погодных условий из ответа погодного API
conditions = {
    'clear': 'ясно',
    'partly-cloudy': 'облачно с прояснениями',
    'cloudy': 'облачно',
    'overcast': 'пасмурно',
    'drizzle': 'моросящий дождь',
    'light-rain': 'мелкий дождь',
    'rain': 'дождь',
    'moderate-rain': 'умеренный дождь',
    'heavy-rain': 'сильный дождь',
    'continuous-heavy-rain': 'продолжительный сильный дождь',
    'showers': 'ливень',
    'wet-snow': 'мокрый снег',
    'light-snow': 'слабый снег',
    'snow': 'снег',
    'snow-showers': 'снегопад',
    'hail': 'град',
    'thunderstorm': 'гроза',
    'thunderstorm-with-rain': 'дождь с грозой',
    'thunderstorm-with-hail': 'гроза с градом'
}

# Проверка полученного сообщения на русский язык, ищем из 5
# верхних вариантов, т.к. иначе бывают ошибки определения
class IsRussian(filters.MessageFilter):
    def filter(self, message):
        return 'ru' in list(zip(*langid.rank(message.text)[:5]))[0]


# Проверка, является ли сообщение запросом о погоде, в ней храним
# название населённого пункта без окончания
class IsWeatherIntent(filters.MessageFilter):
    def __init__(self):
        super().__init__()
        self.location = None
        self.key_words = ['температур', 'погод', 'холодн', 'тепл', 'градус']

    def filter(self, message):

        for word in self.key_words:
            if word in message.text.lower():
                break
        else:
            return False

        processed_text = nlp(message.text)

        for ent in processed_text.ents:
            if ent.label_ == 'LOC':
                break
        else:
            return False

        self.location = ent.text[:-1]
        return True


# Проверка на вопрос из базы. Значение удалённости от ближайшего
# эмбеддинга сравнивается с пороговым значением. Сохраняется
# соответствующий ответ из датасета.
class IsQAIntent(filters.MessageFilter):
    def __init__(self):
        super().__init__()
        self.answer = None

    def filter(self, message):
        sentence = embed_sentence(message.text)
        idx, distance = annoy_index.get_nns_by_vector(sentence, 1, include_distances=True)
        if distance[0] > 0.45:
            return False
        self.answer = df['answer_text'].iloc[idx[0]]
        return True


# Функция для создания эмбеддинга предложения методом mean pooling.
# Используется для qa-интента для текста вопроса
def embed_sentence(sentence):
    text = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        model_output = model(**text)

    token_embeddings = model_output[0]
    expanded_mask = text['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

    return (torch.sum(token_embeddings * expanded_mask, dim=1) /
            torch.clamp(expanded_mask.sum(dim=1), min=1e-9)).squeeze()


# Функция для определения координат населённого пункта по названию.
# Используем API
def get_coords(place):

    querystring = {'query': place, 'count': 5}
    response = requests.request('GET', coords_url, headers=coords_headers, params=querystring)

    for result in response.json()['suggestions']:
        if result['data']['geo_lat'] is not None:
            return result['value'], result['data']['geo_lat'], result['data']['geo_lon']


# Функция для получения данных о погоде из координат населённого пункта.
# Используем API
def get_weather(coords):
    if coords is None:
        return 'К сожалению, данного населённого пункта нет в базе.'

    location_full, lat, lon = coords
    querystring = {'lat': lat, 'lon': lon, 'lang': 'ru_RU'}

    response = requests.request('GET', weather_url, headers=weather_headers, params=querystring)

    if response.status_code == 403:
        return 'Нет доступа к погоде. Возможно, превышен лимит запросов за сутки.'

    if response.status_code != 200:
        return 'Ошибка выполнения запроса.'

    weather_info = response.json()['fact']

    return f'В ({location_full}) сейчас {weather_info["temp"]} градусов Цельсия, ' \
           f'{conditions[weather_info["condition"]]}, а атмосферное давление ' \
           f'составляет {weather_info["pressure_mm"]} мм. рт. ст.'


# Функция для генерации текста в разговорном интенте. Перед этим добавляет
# специальные токены и делает токенизацию. Держит в памяти контекст определённой
# длины
def generate_text(text):
    global chat_context
    bos = '[BOS] ' if not chat_context else ''
    text = bos + text + ' [SEP]'
    chat_context.append(text)
    chat_context = chat_context[-8:]
    text = ' '.join(chat_context)
    text = tokenizer_chat(text, return_tensors='pt')
    text = {k: v.to(model_chat.device) for k, v in text.items()}
    prefix_len = len(text['input_ids'][0])

    generated = model_chat.generate(**text,
                                    bos_token_id=tokenizer_chat.bos_token_id,
                                    pad_token_id=tokenizer_chat.pad_token_id,
                                    eos_token_id=tokenizer_chat.sep_token_id,
                                    do_sample=True,
                                    temperature=1.5,
                                    num_beams=10,
                                    top_k=20,
                                    max_new_tokens=256,
                                    repetition_penalty=3.,
                                    no_repeat_ngram_size=3,
                                    # length_penalty=1.5,
                                    num_return_sequences=1)

    decoded = tokenizer_chat.decode(generated[0][prefix_len:], skip_special_tokens=True).strip()
    chat_context.append(decoded + ' [SEP]')
    return decoded


# Стартовая функция, очищает контекст беседы
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_context
    chat_context.clear()
    greeting = 'Привет! Я - чат-бот курсового проекта по NLP. Могу рассказать о текущей погоде в российском городе, '\
               'ответить на ряд интересующих вас вопросов (у меня их 3000 в базе), либо можем просто поболтать :) '\
               'Я немного запоминаю контекст нашей беседы, поэтому чтобы начать разговор с чистого листа, снова '\
               'введите команду /start'

    await context.bot.send_message(chat_id=update.effective_chat.id, text=greeting)


# Сообщение, если не распознан русский язык
async def not_russian(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Пожалуйста, напишите по-русски.')


# Функция для вывода сообщения о погоде
async def weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=get_weather(get_coords(is_weather.location)))


# Функция для вывода ответа на вопрос из базы
async def qa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=is_qa.answer)


# Запускаем разговорный интент бота при прохождении других фильтров
async def message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=generate_text(update.message.text))


# Функция для вывода сообщения при получении незнакомой команды
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Извините, незнакомая команда.')


if __name__ == '__main__':

# Интент 1: Погода.
# Модель для задачи NER. Находим населённый пункт в тексте

    nlp = spacy.load('ru_core_news_sm')

# Интент 2: Вопросно-ответная система.
# Загружаем датасет, из которого будем брать ответы, токенизатор, модель
# для создания эмбеддингов вопросов пользователя и модель annoy для поиска
# ближайших вопросов из базы

    df = pd.read_csv('./data/qa_data.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    annoy_index = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index.load('./models/qa_annoy_model.ann')

# Интент 3: Разговорный бот.
# Загружаем нашу дообученную модель и её токенизатор

    chat_context = []
    model_path = './models/gpt/'
    tokenizer_chat = AutoTokenizer.from_pretrained(model_path)
    model_chat = AutoModelForCausalLM.from_pretrained(model_path + 'model').to(device)

# Запускаем бота

    is_russian = IsRussian()
    is_weather = IsWeatherIntent()
    is_qa = IsQAIntent()

    application = ApplicationBuilder().token(f'{bot_token}').build()

    start_handler = CommandHandler('start', start)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    language_handler = MessageHandler(~is_russian, not_russian)
    weather_handler = MessageHandler(is_weather, weather)
    qa_handler = MessageHandler(is_qa, qa)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), message)

    application.add_handler(start_handler)
    application.add_handler(unknown_handler)
    application.add_handler(language_handler)
    application.add_handler(weather_handler)
    application.add_handler(qa_handler)
    application.add_handler(message_handler)

    application.run_polling()
