# AGENTS.md — инструкции для агентов

Репозиторий: Python-пакет `gigaam-v3` — распознавание русской речи (ASR)
моделью [GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) (Conformer ~220M,
вариант e2e-rnnt) поверх ONNX Runtime.

## Что делает проект

- `src/gigaam/model.py` — `GigaAM`: три ONNX-сессии (encoder/decoder/joint)
  + greedy RNN-T декодер. `transcribe_path()`: файл → текст.
- `src/gigaam/features.py` — log-mel фейчуризер (порт onnx-asr
  GigaamPreprocessorNumpy v3): 16 kHz, n_fft=320, hop=160, 64 mel, без CMVN.
  Оконная функция и mel-банк — бинарные ассеты из
  `src/gigaam/data/gigaam_features_assets.npz`.
- `src/gigaam/models.py` — скачивание ONNX-файлов с Hugging Face
  (`istupakov/gigaam-v3-onnx`) при первом запуске в `./models`
  (env `GIGAAM_MODELS_DIR` / флаг `--models-dir`). Модель из пакета исключена.
- `src/gigaam/cli.py` — entry point `gigaam` (аудио → текст).
- `src/gigaam/server.py` — entry point `gigaam-server` (FastAPI, POST
  `/v1/audio/transcriptions` и `/inference`) для Vocalinux `remote_api`.

Entry points объявлены в `pyproject.toml` (`[project.scripts]`). Пакет
поставлется через `uvx --from . gigaam ...` (или `--from git+https://...`).
Зависимости: numpy, onnxruntime, fastapi, uvicorn, python-multipart.

## Команды

```bash
# dev-окружение
uv venv .venv --python 3.14
uv pip install --python .venv -e .

# транскрибация (модель подтянется в ./models)
.venv/bin/gigaam --time <audio>

# сервер
.venv/bin/gigaam-server --port 8000
```

Проверка без установки: `uvx --from . gigaam <audio>`.

Тестового аудио/набора тестов в репозитории нет — быстрая проверка:
сгенерировать русскую речь `espeak-ng -v ru -s 120 -w /tmp/t.wav "текст"`,
затем прогнать `gigaam /tmp/t.wav`. Известная хорошая фраза и ожидаемый
результат, по которому сверяется пайплайн:
«Это текст для проверки работы системы распознавания русской речи на основе.».

## Важные ограничения (не «баги»)

- **Vulkan-инференса нет.** Vulkan EP удалён из официального onnxruntime
  (ещё в v1.15, заменён WebGPU EP); пакета `onnxruntime-vulkan` не существует.
  Быстрые пути: CPU (21× realtime на Ryzen 9950X) или CUDA через
  `onnxruntime-gpu`.
- blank-токен = id **1024** (`<blk>`), vocab из 1025 токенов SentencePiece.
- Энкодер требует `audio_signal` [batch,64,T] f32 и `length` [batch] i64,
  отдаёт `encoded` [batch,768,T/4] и `encoded_len` i32.
- `features.load_audio` умеет только путь, на входе — ffmpeg (s16le 16k mono).
  Для байтового WAV есть `server.wav_to_waveform`.
- Синтетика espeak-ng распознаётся хуже живой речи — это аудио, не код.

## Правила

- Не коммитить в git: `models/`, `*.onnx`, `*.dmg`, `.venv/`
  (см. `.gitignore`).
- `models/` — это сгенерированный кэш (скачивается с HF); не править,
  не добавлять версии.
- Изменения фейчуризера/декодера сверить с эталоном: JS
  `src/workers/onnxWorker.js` из приложения Type (greedy loop: lastToken,
  h/c state, MAX_TOKENS_PER_STEP=3) и GigaamPreprocessorNumpy v3.
- Код без комментариев-«простыней», стиль — как в существующих модулях;
  клиентские флаги дублируют дефолты из `model.py` (`DEFAULT_PROVIDERS`).