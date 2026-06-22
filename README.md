# Aether

Aether — автономный CLI-агент для написания и рефакторинга кода через Ollama.

```bash
aether "Напиши тесты для модуля parser.py"
aether --init          # первый запуск: выбор модели Ollama
aether --help
```

## Возможности

- **Нативный tool calling** через Ollama (`ollama.chat(tools=...)`) — модель сама решает, читать файл, писать, запускать shell или искать код
- **Единый путь выполнения**: tool calling — основной, текстовый fallback при ошибке API
- **edit_file с diff-first**: точечные замены через `[{old, new}]`, полная перезапись только для новых файлов или >50% изменений
- **Подтверждение опасных действий**: запрос подтверждения перед `write_file`, `edit_file`, `run_shell` (отключается через `--yes`)
- **Контекст проекта**: AST-парсинг .py-файлов для анализа импортов и структуры
- **Simple chat detection**: короткие сообщения определяются как диалог без вызова инструментов
- **Bilingual**: промпты на русском и английском
- **Умная память**: логирование действий в `aether_memory.json` с поиском по ключевым словам

## Установка

```bash
git clone https://github.com/0penAGI/aether.git
cd aether
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./install.sh
```

После `install.sh` команда `aether` будет доступна глобально (symlink в `~/.local/bin`).

## Использование

```bash
# Первый запуск — выбор модели
aether --init

# Выполнить задачу
aether "Создай файл hello.py с функцией приветствия"

# Пропустить подтверждения
aether --yes "Отрефактори main.py, вынеси логику в отдельные функции"

# Указать модель (переопределяет сохранённую в конфиге)
aether --model qwen2.5-coder "Напиши юнит-тесты"
```

### Флаги

| Флаг | Описание |
|------|----------|
| `--init` | Первоначальная настройка (выбор модели Ollama) |
| `--model`, `-m` | Модель Ollama (переопределяет конфиг) |
| `--yes`, `-y` | Пропустить все подтверждения |
| `task` | Задача для выполнения (позиционный аргумент) |

## Тестирование

```bash
# unit-тесты (52 теста)
pytest test_aether.py -v

# интеграционные тесты (требуют запущенный Ollama)
pytest test_aether_integration.py -v

# все тесты
pytest test_aether.py test_aether_integration.py -v
```

## Зависимости

- [Ollama](https://ollama.ai) — запущенный сервер с поддержкой tool calling
- Python 3.10+
- `ollama`, `rich`, `numpy`, `faiss`, `playwright`, `requests` (см. `requirements.txt`)

## Структура

```
├── aether.py               # Основной агент (2836 строк, 112 методов, 9 классов)
├── aether_config.py        # CLI-вход, конфиг, first-run wizard
├── bin/aether              # bash-обёртка для запуска
├── test_aether.py          # 52 unit-теста
├── test_aether_integration.py  # 13 интеграционных тестов
├── install.sh              # Установка в ~/.local/bin
└── requirements.txt        # Зависимости
```

## Лицензия

MIT
