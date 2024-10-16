from openai import OpenAI, AuthenticationError, Completion, NotFoundError
import configparser
import requests

KEYS_PATH = '../.secrets/keys.ini'
DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_openai_key(keys_path: str):
    config = configparser.ConfigParser()
    config.read(keys_path)
    openai_api_key = config['openai']['API_KEY']
    openai_base = _url = config['openai']['BASE_URL']
    return openai_api_key, openai_base


def get_limits(api_key: str):
    url = 'https://openrouter.ai/api/v1/auth/key'
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    print(response.status_code)
    print(response.content)


def summarize(source_text,
              api_key: str,
              base_url: str,
              model: str,
              tags_quantity: int = 3) -> Completion:
    defailt_model = DEFAULT_MODEL
    client = OpenAI(api_key=api_key, base_url=base_url)  # empty = env variable OPENAI_API_KEY
    system = [{"role": "system", "content": "You are Summary AI."}]
    user = [{"role": "user",
             "content": f"""Briefly outline the original document in Russian; as a result, at least one sentence 
             should be obtained from each original paragraph; it is also advisable to use numeric values; also return 
             no more than {tags_quantity} tags that correspond to the content of the document. Return the result in JSON format 
             using the following structure: "result": "Summarized text", "tags": ["example", "text", "summary"] 
             Source document:\n\n{source_text}"""}]
    chat_history = []  # past user and assistant turns, for AI memory
    try:
        result = client.chat.completions.create(
            messages=system + chat_history + user,
            model=model,
            max_tokens=500, top_p=0.4, temperature=0.3,
        )
        return result
    except AuthenticationError:
        print('API key credit limit reached.')
    except NotFoundError:
        if not model == defailt_model:
            print(f'Model {model} not found. Trying default {defailt_model}.')
            return summarize(source_text,
                             api_key,
                             base_url,
                             defailt_model,
                             tags_quantity)
        else:
            print(f'Default model {defailt_model} not found.')
            return None


if __name__ == '__main__':
    api_key, base_url = get_openai_key(KEYS_PATH)
    text = """Никелевые руды Никелевые руды — вид полезных ископаемых, природные минеральные образования, содержание 
    никеля в которых достаточно для экономически выгодного извлечения этого металла или его соединений. Обычно 
    разрабатываются месторождения сульфидных руд, содержащие 1—2 % Ni, и силикатные руды, содержащие 1—1,
    5 % Ni. К наиболее важным минералам никеля относят наиболее часто встречающиеся и промышленные минералы: сульфиды 
    (пентландит (Fe, Ni)S (или (Ni, Fe)9S8; содержит 22—42 % Ni), миллерит NiS (64,5 % Ni), никелин (купферникель, 
    красный никелевый колчедан) NiAs (до 44 % Ni), никелистый пирротин, полидимит, кобальт-никелевый пирит, виоларит, 
    бравоит, ваэсит NiS2, хлоантит, раммельс-бергит NiAs2, герсдорфит (герсфордит, никелевый блеск NiAsS), ульманит), 
    водные силикаты (гарниерит, аннабергит, ховахсит, ревдинскит, шухардит, никелевые нонтрониты) и никелевые хлориты.
    """
    result = None
    result = summarize(text, api_key, base_url)
    try:
        result_text = result.choices[0].message.content
        source_tokens = result.usage.prompt_tokens
        result_tokens = result.usage.completion_tokens
        print(result_text, f'\n{source_tokens}\n{result_tokens}')
    except AttributeError:
        print('Получен пустой ответ.')

    get_limits(api_key)
