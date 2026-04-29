# Semantic Steganography with YOLO + SVD

Progetto per esperimenti di steganografia semantica su immagini COCO basato su:

- rilevamento oggetti con YOLO pretrained;
- selezione di una ROI semantica;
- embedding del payload tramite SVD implementata senza usare `np.linalg.svd`;
- estrazione `non_blind` e baseline `blind`;
- valutazione con attacchi, metriche e salvataggio strutturato dei risultati.

## Obiettivi

La pipeline permette di confrontare:

- strategie ROI: `largest`, `smallest`, `random`, `full_image`;
- bande SVD: `high_energy`, `mid_energy`, `low_energy`;
- decoder: `non_blind`, `blind`;
- attacchi: `none`, `gaussian_noise`, `gaussian_blur`, `jpeg_compression`.

Il focus del progetto e' su modularita', riproducibilita' e facilita' di sperimentazione.

## Struttura del progetto

```text
semantic_stego/
  config/
  data/
  detection/
  svd/
  stego/
  attacks/
  metrics/
  experiments/
  cli/
main.py
requirements.txt
tests/
scripts/
```

## Moduli principali

### `semantic_stego/config`

- `defaults.py`: default globali e configurazione debug minima
- `cli_args.py`: definizione completa degli argomenti CLI
- `schemas.py`: dataclass per config, detection, ROI, metadata e risultati intermedi

### `semantic_stego/data`

- `coco_loader.py`: iterazione riproducibile delle immagini COCO da `data/coco/<split>`
- `image_io.py`: lettura/scrittura immagini, crop/paste ROI, conversioni colore e tipo

### `semantic_stego/detection`

- `yolo_detector.py`: wrapper YOLO pretrained con output normalizzato in `Detection`
- `roi_selector.py`: selezione ROI secondo la strategia scelta

### `semantic_stego/svd`

- `svd_from_scratch.py`: decomposizione SVD basata su eigendecomposizione simmetrica
- `svd_utils.py`: selezione indici singolari e reconstruction error

### `semantic_stego/stego`

- `payload.py`: conversione testo/bit, generazione casuale e gestione capacita'
- `embedder.py`: embedding del payload sul canale `Y` in spazio `YCrCb`
- `extractor.py`: estrazione `non_blind` e `blind`

### `semantic_stego/attacks`

- `attacks.py`: applicazione uniforme di rumore, blur e JPEG

### `semantic_stego/metrics`

- `image_metrics.py`: PSNR e SSIM full/ROI
- `message_metrics.py`: BER, bit errors, exact match, character accuracy
- `timing.py`: timer semplice in millisecondi

### `semantic_stego/experiments`

- `grid.py`: costruzione della griglia attacchi
- `runner.py`: orchestrazione end-to-end della pipeline
- `result_writer.py`: scrittura di `config.json`, `results.csv`, `results.jsonl`, `failures.jsonl`, `summary.csv`

### `semantic_stego/cli`

- `app.py`: entrypoint CLI che costruisce `ExperimentConfig` e avvia il runner

## Requisiti

Installazione dipendenze:

```bash
pip install -r requirements.txt
```

Dipendenze richieste:

- `numpy`
- `opencv-python`
- `Pillow`
- `scikit-image`
- `pandas`
- `tqdm`
- `ultralytics`
- `pytest`

## Dataset atteso

La pipeline assume una struttura tipo:

```text
data/
  coco/
    val2017/
      000000000139.jpg
      ...
```

## Esecuzione rapida

### Avvio con `main.py`

Usa la configurazione debug di default:

```bash
python main.py
```

### Avvio tramite CLI

Esempio debug su 10 immagini:

```bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/debug \
  --max-images 10 \
  --yolo-model yolov8n.pt \
  --roi-strategies largest full_image \
  --svd-bands mid_energy \
  --decoders non_blind \
  --attacks none \
  --payload-bits 64 \
  --embedding-strength 10 \
  --seed 42 \
  --save-roi-debug
```

Esempio ablation ridotta:

```bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/coco_ablation_small \
  --max-images 50 \
  --yolo-model yolov8n.pt \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg \
  --noise-sigmas 5 \
  --blur-kernels 3 \
  --jpeg-qualities 90 \
  --payload-bits 128 \
  --embedding-strength 10 \
  --seed 42
```

## Opzioni CLI principali

- `--coco-root`: root del dataset COCO
- `--split`: split immagini, tipicamente `val2017`
- `--output-dir`: directory output della run
- `--max-images`: massimo numero immagini da processare
- `--image-size`: dimensione inferenza YOLO
- `--yolo-model`: path o nome del modello YOLO
- `--confidence-threshold`: soglia confidence detection
- `--roi-strategies`: lista strategie ROI
- `--svd-bands`: lista bande SVD
- `--decoders`: lista decoder
- `--attacks`: lista attacchi
- `--noise-sigmas`: sigma per rumore gaussiano
- `--blur-kernels`: kernel per blur gaussiano
- `--jpeg-qualities`: quality JPEG
- `--payload-text`: payload testuale opzionale
- `--payload-bits`: lunghezza payload in bit
- `--payload-seed`: seed del payload casuale
- `--embedding-strength`: forza di embedding / delta QIM
- `--seed`: seed globale per campionamento e ROI random
- `--min-roi-area`: area minima ROI valida
- `--payload-policy`: `truncate_message`, `skip_image`, `raise_error`
- `--skip-no-detection` / `--no-skip-no-detection`: gestione immagini senza detection
- `--save-images`: salva immagini stego
- `--save-roi-debug`: salva immagini con ROI disegnata

## Output della pipeline

Ogni run scrive in `outputs/<run_name>/`:

```text
config.json
results.csv
results.jsonl
failures.jsonl
summary.csv
roi_debug/
images/
```

Note:

- `failures.jsonl` viene creato solo se ci sono errori o configurazioni fallite
- `images/` viene creato solo con `--save-images`
- `roi_debug/` viene creato solo con `--save-roi-debug`

## Metriche salvate

La pipeline salva, tra le altre:

- `PSNR_full`, `PSNR_roi`
- `SSIM_full`, `SSIM_roi`
- `bit_errors`, `BER`, `exact_match`
- `payload_bits_requested`, `payload_bits_capacity`, `payload_bits_embedded`, `payload_bits_dropped`
- `payload_retention_ratio`, `payload_success_ratio`
- `bpp_roi`, `bpp_image`
- `yolo_time_ms`, `svd_time_ms`, `embedding_time_ms`, `extraction_time_ms`, `attack_time_ms`, `total_time_ms`
- `svd_reconstruction_error`

## YOLO weights

Il file dei pesi, ad esempio `yolov8n.pt`, non viene versionato ed e' ignorato da git tramite `*.pt`.

Puoi passare alla CLI:

- un path locale, per esempio `--yolo-model yolov8n.pt`
- oppure un nome modello gestito da Ultralytics, se vuoi lasciare il download al runtime

## Test

Esecuzione test:

```bash
pytest
```

Coprono almeno:

- selezione ROI
- payload
- SVD
- embedding/extraction
- attacchi
- metriche
