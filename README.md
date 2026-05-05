# Semantic Steganography with YOLO + SVD

Progetto per esperimenti di steganografia semantica su immagini COCO basato su:

- rilevamento oggetti con YOLO pretrained;
- selezione di una ROI semantica;
- embedding del payload tramite SVD implementata senza usare `np.linalg.svd`;
- estrazione `non_blind` coerente con il QIM di embedding e baseline `blind`;
- valutazione con attacchi, metriche e salvataggio strutturato dei risultati.

## Obiettivi

La pipeline permette di confrontare:

- strategie ROI: `largest`, `smallest`, `random`, `full_image`;
- bande SVD: `high_energy`, `mid_energy`, `low_energy`;
- decoder: `non_blind`, `blind`;
- attacchi: `none`, `gaussian_noise`, `gaussian_blur`, `jpeg_compression`.

Il focus del progetto e' su modularita', riproducibilita' e facilita' di sperimentazione.

Protocollo consigliato per confronti fair tra ROI:

- usare `--payload-text` con una stringa fissa condivisa da tutte le configurazioni;
- distinguere tra run ottimizzati e run di confronto completo;
- nel caso di `payload_policy=truncate_message`, accettare un'immagine se esiste almeno una configurazione ROI/banda embeddabile e registrare come failure isolate le combinazioni incompatibili.

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
- `extractor.py`: estrazione `non_blind` allineata al QIM e baseline `blind`

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
- `matplotlib`
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
.venv/bin/python main.py
```

### Avvio tramite CLI

Esempio debug su 10 immagini:

```bash
.venv/bin/python -m semantic_stego.cli.app \
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
  --repetition-factor 3 \
  --seed 42 \
  --save-roi-debug
```

Esempio evaluation completa via CLI:

```bash
.venv/bin/python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/evaluation_50 \
  --max-images 50 \
  --yolo-model yolov8n.pt \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg_compression \
  --noise-sigmas 5 \
  --blur-kernels 3 \
  --jpeg-qualities 90 \
  --payload-text "SEMANTIC_STEGO_TEST" \
  --embedding-strength 20 \
  --repetition-factor 3 \
  --seed 42
```

Con `--payload-text`, la pipeline usa sempre la stessa stringa per tutte le configurazioni. `--max-images` indica il numero massimo di immagini accettate, non semplicemente lette dal dataset.

### Script consigliati

- `scripts/run_evaluation.sh <N> <payload>`: profilo ottimizzato per ottenere grafici piu' leggibili. Di default usa `roi_strategy=largest`, `svd_band=high_energy`, `embedding_strength=20`, `repetition_factor=3`, lancia anche l'analisi finale e apre la cartella di output.
- `scripts/run_full_comparison.sh <N> <payload>`: griglia completa per confrontare il ruolo di YOLO (`largest`, `smallest`, `random`) contro la baseline senza YOLO (`full_image`) usando tutte le bande SVD, i decoder e gli attacchi.
- `scripts/run_clean_sweep.sh <N> <payload>`: sweep clean-only per confrontare `embedding_strength` e `repetition_factor` senza attacchi e scegliere i parametri migliori prima del benchmark completo.

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
- `--payload-text`: payload testuale fisso consigliato per confronti fair tra ROI
- `--payload-bits`: lunghezza payload in bit
- `--payload-seed`: seed del payload casuale
- `--embedding-strength`: forza di embedding / delta QIM
- `--repetition-factor`: numero di ripetizioni per il codice di maggioranza usato in embedding/extraction
- `--seed`: seed globale per campionamento e ROI random
- `--min-roi-area`: area minima ROI valida
- `--payload-policy`: `truncate_message`, `skip_image`, `raise_error`
- `--skip-no-detection` / `--no-skip-no-detection`: gestione immagini senza detection
- `--save-images`: salva immagini stego
- `--save-roi-debug`: salva immagini con ROI disegnata

Note operative su payload e compatibilita':

- con `--payload-text`, il payload viene convertito in bit una sola volta e riusato in tutte le configurazioni;
- con `payload_policy=truncate_message`, un'immagine viene processata se esiste almeno una configurazione ROI/banda con capacita' utile; le combinazioni incompatibili vengono comunque registrate in output con `status=failed_payload_incompatible`;
- con policy piu' restrittive come `skip_image` o `raise_error`, la capacita' richiesta resta il payload completo;
- le immagini completamente scartate vengono tracciate in `results.csv` e `failures.jsonl` con `status=failed_payload_incompatible`.

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

- `image_accepted`, `image_filter_reason`
- `PSNR_full`, `PSNR_roi`
- `SSIM_full`, `SSIM_roi`
- `bit_errors`, `BER`, `exact_match`
- `payload_bits_requested`, `payload_bits_capacity`, `payload_bits_embedded`, `payload_bits_dropped`
- `payload_retention_ratio`, `payload_success_ratio`
- `bpp_roi`, `bpp_image`
- `yolo_time_ms`, `svd_time_ms`, `embedding_time_ms`, `extraction_time_ms`, `attack_time_ms`, `total_time_ms`
- `svd_reconstruction_error`

## Analisi risultati

Per aggregare una o piu' run, generare ranking automatici, conclusioni finali e grafici comparativi:

```bash
.venv/bin/python scripts/analyze_results.py outputs/evaluation_50
```

Per confrontare piu' run nello stesso report:

```bash
.venv/bin/python scripts/analyze_results.py outputs/run_a outputs/run_b --analysis-dir outputs/analysis/full_tests
```

Se non passi path espliciti, lo script prova a scoprire automaticamente tutte le run sotto `outputs/`.

Artifact prodotti in `outputs/analysis/` o nella directory specificata con `--analysis-dir`:

- `consolidated_results.csv`: merge completo di tutte le run
- `embedding_summary.csv`: confronto per embedding config (`roi_strategy`, `svd_band`)
- `extraction_summary.csv`: confronto per configurazione di estrazione
- `attack_summary.csv`: confronto per configurazione e attacco
- `category_summary.csv`: vista macro per ROI, banda, decoder e attacco
- `analysis_overview.json`: best config in formato machine-readable
- `conclusions.md`: conclusioni finali testuali
- `plots/`: ranking, scatter plot e heatmap comparative

L'analisi aggregata distingue ora anche:

- `exact_match_rate_all`: exact match su tutte le run di successo;
- `exact_match_rate_complete`: exact match limitato ai casi con payload completo, senza troncamento;
- `complete_payload_rate`: quota di run che hanno embeddato tutto il payload richiesto;
- misure di variabilita' di base come `std` e `sem` per supportare la lettura dei grafici.

## Costi computazionali SVD

Per ROI dense e quasi quadrate, sia la SVD custom sia `np.linalg.svd` hanno costo asintotico cubico in funzione della dimensione della matrice. La differenza pratica e' nelle costanti:

- la pipeline custom costruisce prima la matrice di Gram (`A^T A` oppure `A A^T`);
- risolve poi una eigendecomposizione simmetrica con `np.linalg.eigh`;
- completa infine con un passaggio esplicito di ortonormalizzazione delle colonne.

Questo rende la variante custom piu' lenta della SVD standard, ma e' una scelta intenzionale: la pipeline principale privilegia una SVD didattica/autonoma coerente con i vincoli progettuali del repository, mentre `np.linalg.svd` resta solo una baseline di riferimento nei test.

Benchmark locale su 20 matrici casuali `64 x 64`, media su 5 esecuzioni per matrice. L'errore riportato e' l'errore relativo di ricostruzione `||A - USV^T|| / ||A||`.

| Metodo                        | Tempo medio SVD | Dev. std | Errore ricostruzione | Note                |
| ----------------------------- | --------------: | -------: | -------------------: | ------------------- |
| SVD custom/eigendecomposition |         6.10 ms |  0.50 ms |            1.24e-14 | pipeline principale |
| `np.linalg.svd` / standard    |         0.66 ms |  0.05 ms |            2.13e-15 | baseline solo test  |

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
