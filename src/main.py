from __future__ import annotations

import shutil
import json
import os
import sys
import configparser

from shutil import move
from typing import Dict, List, Any
from pathlib import Path
from docx import Document
from text_chunker import TextChunker
from fileman import prepare_filename
from open_ai import summarize

if sys.version_info < (3, 10):
    raise RuntimeError("This package requires Python 3.10+")

KEYS_PATH = '../.secrets/keys.ini'
DOCS_PATH = '../docs'
RESULTS_PATH = '../results'
ERROR_ANSWER = '\nПроблема в работе скрипта. Обратитесь к разработчику.'


def prepare_absolute_path(relative_path) -> Path:
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    if '..' in str(relative_path):
        return dir_path / Path(relative_path)
    else:
        return Path(relative_path)


def get_files(dir_name: str, file_types: list | None = None) -> list:
    if file_types is None:
        file_types = ['docx']
    directory = prepare_absolute_path(dir_name)
    result = []
    for next_file in directory.iterdir():
        filename = next_file.name
        if any([filename.endswith(end) for end in file_types]):
            result.append(os.path.join(os.fsdecode(directory), filename))
    return result


def parse_answer(text) -> json:
    if '{' in text and '}' in text:
        text = '{' + text.split("{")[1].rsplit('}')[0] + '}'
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = None
    else:
        result = None
    return result


def get_text_from_docx(file_path: Path | str) -> str:
    pars = [paragraph.text for paragraph in Document(file_path).iter_inner_content()]
    return '\n'.join(pars)


def get_openai_key(keys_path: str) -> tuple:
    config = configparser.ConfigParser()
    config.read(prepare_absolute_path(keys_path))
    openai_api_key = config['openai']['API_KEY']
    openai_base = _url = config['openai']['BASE_URL']
    return openai_api_key, openai_base


def prepare_json(answer, result_json: Dict[str, Any]) -> None:
    result_string = answer.choices[0].message.content
    tmp_json = parse_answer(result_string)
    match tmp_json:
        case tmp_json if tmp_json is not None and isinstance(tmp_json, dict):
            result_json['summary_text'] += tmp_json['result']
            result_json['tags'].extend(tmp_json['tags'])
            result_json['usage']['input'] += answer.usage.prompt_tokens
            result_json['usage']['output'] += answer.usage.completion_tokens
            result_json['model'] = answer.model
        case _:
            result_json['summary_text'] += ERROR_ANSWER


def repair_tags(tags: List[str]) -> List[str]:
    result = []
    for i in tags:
        result.extend(i.split(' '))
    return list(set(result))


def process_file(document: Path | str,
                 ai_model: str | None = None,
                 max_token_quantity: int = 8000) -> Dict[str, Any]:
    if ai_model is None:
        ai_model = "openai/gpt-4o-mini"
    text = get_text_from_docx(document)
    result_json = {
        "file_name": Path(document).name,
        "summary_text": '',
        "tags": [],
        "model": ai_model,
        "usage": {
            "input": 0,
            "output": 0}
    }
    openai_key, base_url = get_openai_key(KEYS_PATH)
    chunker = TextChunker(maxlen=max_token_quantity * 3)
    for chunk in chunker.chunk(text):
        answer = summarize(chunk,
                           api_key=openai_key,
                           base_url=base_url,
                           model=ai_model,
                           tags_quantity=1)
        if answer is not None:
            prepare_json(answer, result_json)
            move_to_arch(document)
        else:
            result_json["summary_text"] += ERROR_ANSWER
    result_json["tags"] = repair_tags(list(set(result_json["tags"])))
    return result_json


def save_result(result_json: dict, result_path: str = RESULTS_PATH):
    result_path = prepare_absolute_path(result_path)
    source_filename = '.'.join(result_json["file_name"].split('.')[:-1])
    filename = prepare_filename(source_filename,
                                'json')[0]
    result_filename = result_path / filename
    with open(result_filename, 'w') as json_file:
        json_file.write(json.dumps(result_json,
                                   indent=2,
                                   ensure_ascii=False))


def find_filename(search_path: str | Path, file_name_part: str) -> Path | None:
    results_path = prepare_absolute_path(search_path)
    try:
        return [filename for filename in results_path.iterdir() if file_name_part in filename.name][0]
    except IndexError:
        return None


def move_to_arch(file_for_move: Path | str):
    if file_for_move is not None:
        file_for_move = Path(str(file_for_move))
        archive_path = file_for_move.parent / 'archive'
        if not archive_path.exists():
            os.mkdir(archive_path)
        try:
            move(str(file_for_move), archive_path)
            print(f'file {file_for_move} moved')
        except shutil.Error:
            file_for_remove = archive_path / file_for_move.name
            os.remove(file_for_remove)
            move_to_arch(file_for_move)


def conc_jsons(results_path: str | Path) -> dict:
    results_path = prepare_absolute_path(results_path)
    result_file_name = 'результат успешных суммаризаций'
    suffix = '.json'
    result_file = find_filename(results_path, result_file_name)
    if result_file is None:
        result_json = {'file_name': result_file_name + suffix,
                       'results': {}}
    else:
        with open(result_file, 'r') as f:
            result_json = json.loads(f.read())
    for next_file in sorted(results_path.iterdir()):
        try:
            result_file_name = result_file.name
        except AttributeError:
            result_file_name = ''
        if next_file.is_file() and next_file.suffix == suffix and next_file.name != result_file_name:
            with open(next_file, 'r') as f:
                next_json = json.loads(f.read())
                result_json['results'].update({next_json['file_name']: next_json})
            if next_json['file_name'] in result_json['results'].keys():
                move_to_arch(next_file)
    move_to_arch(result_file)
    return result_json


def run(document, model):
    max_tokens_quantity = 8000
    my_json = process_file(document, model, max_tokens_quantity)
    if my_json is not None and isinstance(my_json, dict):
        save_result(my_json)
    else:
        print(ERROR_ANSWER)


if __name__ == '__main__':
    ai_model_name = "openai/gpt-4o-mini"
    for file in get_files(DOCS_PATH):
        run(file, ai_model_name)
    save_result(conc_jsons(RESULTS_PATH), RESULTS_PATH)
