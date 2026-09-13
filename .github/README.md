# gigaam-v3 — русская речь в текст (GigaAM v3, ONNX Runtime)

Локальный распознаватель русской речи на модели
[ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
(Conformer ~220M, вариант **e2e-rnnt** — с пунктуацией и нормализацией текста),
запускается через [ONNX Runtime](https://github.com/microsoft/onnxruntime)
на CPU (или CUDA, см. ниже).

## Быстрый старт

Пакет ставится и запускается через [`uvx`](https://docs.astral.sh/uv/):

```bash
# локально, из каталога проекта:
uvx --from . gigaam speech.wav

# из GitHub:
uvx --from git+https://github.com/badigady/gigaam-v3 gigaam speech.wav
```

### Про модель

**Модель в пакет не входит** — при первом запуске она скачивается
автоматически с Hugging Face
([istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx),
тот же экспорт весов, что использует эталонное приложение):

| файл | размер |
|---|---|
| `v3_e2e_rnnt_encoder.onnx` | ~885 МБ (fp32) |
| `v3_e2e_rnnt_decoder.onnx` | ~4.6 МБ |
| `v3_e2e_rnnt_joint.onnx` | ~2.7 МБ |
| `v3_e2e_rnnt_vocab.txt` | ~13 КБ |

Файлы сохраняются в каталог `./models` рядом с проектом (один раз, дальше
перекачивание не требуется). Каталог можно переопределить переменной
`GIGAAM_MODELS_DIR` или флагом `--models-dir`.

## Использование

```bash
# CLI: аудио → текст (любые входы ffmpeg: wav/mp3/flac/ogg)
gigaam [--time] [--models-dir DIR] [--providers ...] file.wav [file2.mp3 ...]

# HTTP-сервер для Vocalinux (см. ниже)
gigaam-server [--host 127.0.0.1] [--port 8000] [--models-dir DIR]
```

Пример:

```bash
uvx --from . gigaam --time ~/record.wav
```

## Подключение к Vocalinux

Vocalinux умеет распознавать через внешний сервис (движок `remote_api`).
Поднимаем наш сервер и указываем на него:

```bash
uvx --from . gigaam-server        # http://127.0.0.1:8000
```

Сервер отвечает на оба формата:
- `POST /v1/audio/transcriptions` (OpenAI / FunASR),
- `POST /inference` (whisper.cpp).

В `~/.config/vocalinux/config.json`:

```json
"engine": "remote_api",
"remote_api_url": "http://127.0.0.1:8000",
"remote_api_endpoint": "/v1/audio/transcriptions"
```

После правки полностью перезапустите Vocalinux (выйти из трея). То же можно
настроить в GUI: Settings → Speech recognition → Remote API.

## Автостарт через systemd

В репозитории есть юнит-шаблон `systemd/gigaam@.service`: порт берётся из имени
инстанса. Юнит сам соберёт пакет с GitHub через `uvx`, скачает модель и поднимет
сервер без ручных шагов:

```bash
sudo cp systemd/gigaam@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gigaam@8000      # сервер на порту 8000
```

Первый запуск долгий (uvx ставит пакет и зависимости, затем качается модель
~885 МБ), модель сохраняется один раз в `/var/lib/gigaam/models`.
Следить за ходом:

```bash
journalctl -u gigaam@8000 -f
```

Перед активацией проверьте URL репозитория в `Environment=GIGAAM_REPO`
(/etc/systemd/system/gigaam@.service) — он должен указывать на опубликованный
репозиторий с этим пакетом (инстанс `gigaam@PORT` стартует с портом PORT).

```bash
sudo systemctl edit gigaam@8000              # переопределить параметры
sudo systemctl restart gigaam@8000
```

## Производительность

На CPU AMD Ryzen 9 9950X: 32.7 c аудио → ~1.5 c вычислений — **≈21× реалтайм**
(энкодер — основная часть, декодер последовательный ~0.1 c).

## CUDA vs Vulkan

- **CUDA** (RTX 3090): поставьте `onnxruntime-gpu` вместо `onnxruntime`
  и запустите с `--providers CUDAExecutionProvider CPUExecutionProvider`.
- **Vulkan: недоступен.** В официальном `onnxruntime` Vulkan-EP удалён ещё в
  v1.15 (заменён WebGPU EP), пакета `onnxruntime-vulkan` на PyPI нет.
  Под Vulkan GigaAM можно гонять только GGUF-вариантами через
  [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
  (`-DTRANSCRIBE_VULKAN=ON`), но это другой рантайм.

## Структура проекта

```
src/gigaam/
  models.py    автоскачивание модели с Hugging Face в ./models
  model.py     GigaAM: энкодер/декодер/джойнт + greedy RNN-T декод
  features.py  log-mel фичи (порт GigaamPreprocessorNumpy v3),
               эталонные window/mel-банк в data/gigaam_features_assets.npz
  cli.py       entry point gigaam
  server.py    entry point gigaam-server (FastAPI, интерфейс Vocalinux)
pyproject.toml  пакет, entry points
```

Featurizer — точный порт фейчуризера приложения Type (идентичен
`onnx-asr` GigaamPreprocessorNumpy v3): 16 кГц, n_fft=320, hop=160, 64 mel,
энкодер с субсемплингом ×4.

## Примечания

- blank-токен = `<blk>` (id 1024), vocab 1025 токенов SentencePiece.
- На синтетическом голосе espeak-ng результат хуже, чем на живой речи —
  это особенность аудио, а не пайплайна.
- Сокращённый энкодер `v3_e2e_rnnt_encoder.int8.onnx` (~224 МБ) есть там же,
  у istupakov — быстрее на CPU при небольшой потере точности.