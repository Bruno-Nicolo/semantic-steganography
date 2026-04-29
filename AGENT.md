# Piano d'azione per implementazione sistema di steganografia semantica con YOLO + SVD

## 1. Obiettivo del sistema

Realizzare un sistema modulare di **steganografia semantica su immagini** che:

1. utilizza **YOLO pretrained** per individuare oggetti e selezionare una Region of Interest, o ROI;
2. applica un algoritmo di embedding basato su **SVD implementata da zero in Python**;
3. permette di testare diverse strategie di selezione ROI, bande di embedding sulla diagonale dei valori singolari e modalità di estrazione;
4. consente esperimenti riproducibili su **COCO**;
5. espone l'intera pipeline tramite **CLI**, mantenendo parametri e flag separati dalla logica principale;
6. permette anche l'esecuzione tramite un `main.py` semplice, senza obbligare l'uso della CLI.

Il sistema deve privilegiare:

- semplicità architetturale;
- clean code;
- moduli testabili separatamente;
- configurabilità degli esperimenti;
- logging e salvataggio strutturato dei risultati.

---

## 2. Dataset scelto

Usare esclusivamente **COCO**.

### Dataset target

Dataset consigliato:

```text
COCO 2017 validation set
```

Uso previsto:

```text
Debug: 50 immagini
Esperimenti iniziali: 100 immagini
Esperimenti finali: 200 immagini
```

### Requisiti sul dataset

Il sistema deve accettare una struttura dati del tipo:

```text
data/
  coco/
    val2017/
      000000000139.jpg
      ...
```

### Filtro immagini

Per la pipeline con ROI YOLO, processare solo immagini per cui YOLO produce almeno una detection valida.

Parametri consigliati:

```text
confidence_threshold = 0.25 oppure 0.35
iou_threshold = default del modello YOLO
min_roi_area = opzionale
```

Se un'immagine non ha detection valide:

- per `roi_strategy = full_image`, può essere processata comunque;
- per `largest`, `smallest`, `random`, deve essere saltata oppure marcata come `failed_no_detection`.

Il comportamento deve essere configurabile.

---

## 3. Strategie sperimentali richieste

Per ogni immagine COCO, il sistema deve testare le combinazioni tra:

### 3.1 Strategie ROI

Implementare queste quattro strategie:

```text
largest
smallest
random
full_image
```

Descrizione:

1. `largest`: usa la bounding box YOLO con area maggiore.
2. `smallest`: usa la bounding box YOLO con area minore.
3. `random`: sceglie casualmente una bounding box tra quelle valide.
4. `full_image`: usa l'intera immagine come ROI.

Per ogni ROI salvare almeno:

```text
roi_strategy
roi_x1
roi_y1
roi_x2
roi_y2
roi_width
roi_height
roi_area
roi_area_ratio
roi_class_id
roi_class_name
roi_confidence
num_detections
```

Per `full_image`:

```text
roi_class_id = null
roi_class_name = "full_image"
roi_confidence = null
```

### 3.2 Strategie SVD

Applicare l'embedding su tre possibili bande della diagonale dei valori singolari:

```text
high_energy
mid_energy
low_energy
```

Descrizione:

1. `high_energy`: componenti principali che conservano maggiore informazione. Corrispondono ai primi valori singolari.
2. `mid_energy`: componenti centrali nella diagonale.
3. `low_energy`: componenti associati a minore informazione. Corrispondono agli ultimi valori singolari utilizzabili.

La selezione deve essere parametrica.

Esempio:

```python
n = len(S)
payload_len = number_of_bits

if band == "high_energy":
    indices = range(0, payload_len)

elif band == "mid_energy":
    start = max(0, n // 2 - payload_len // 2)
    indices = range(start, start + payload_len)

elif band == "low_energy":
    start = max(0, n - payload_len)
    indices = range(start, n)
```

Il codice deve gestire il caso in cui il payload sia troppo grande per la ROI.

Policy configurabili:

- truncate_message
- skip_image
- raise_error

Default consigliato:

- truncate_message

Motivazione:

Il troncamento permette di valutare anche ROI piccole, evitando che molte configurazioni vengano saltate. Tuttavia, per mantenere confronti corretti tra configurazioni, il sistema deve salvare sempre:

- payload_bits_requested
- payload_bits_capacity
- payload_bits_embedded
- payload_bits_dropped
- payload_retention_ratio
- payload_truncated

Le metriche BER, exact_match e character_accuracy devono essere calcolate solo sui bit effettivamente embeddati. In aggiunta, va calcolata una metrica normalizzata rispetto al payload richiesto:

payload_success_ratio = recovered_correct_bits / payload_bits_requested

dove:

recovered_correct_bits = payload_bits_embedded - bit_errors

Questa metrica penalizza sia errori di estrazione sia troncamento del messaggio.

### 3.3 Strategie di estrazione

Implementare due modalità:

```text
non_blind
blind
```

#### `non_blind`

Estrazione confrontando immagine originale e immagine stego.

È la modalità più semplice e robusta per debug. Può usare informazioni dell'immagine originale, per esempio valori singolari originali o differenze tra valori singolari.

#### `blind`

Estrazione senza immagine di partenza.

Questa modalità deve usare solo:

- immagine stego;
- parametri di embedding;
- eventuale chiave/seme;
- metadata minimi ammessi dal progetto.

Nota importante per il coding agent: il vero blind decoding con SVD non è banale. Implementare una versione iniziale robusta e dichiarata come `blind_baseline`, per esempio basata su quantizzazione dei valori singolari.

Strategia consigliata:

- non modificare i valori singolari con semplice addizione;
- usare una regola di quantizzazione tipo QIM, Quantization Index Modulation.

Esempio concettuale:

```text
bit = 0 -> valore singolare quantizzato su multipli pari di delta
bit = 1 -> valore singolare quantizzato su multipli dispari di delta
```

Estrazione:

```text
bit_hat = round(S_i / delta) % 2
```

Parametri:

```text
embedding_strength / delta
```

La modalità `non_blind` può invece usare confronto diretto contro l'originale.

---

## 4. Attacchi da implementare

Implementare un modulo separato per gli attacchi, con interfaccia uniforme.

Attacchi richiesti:

```text
none
gaussian_noise
gaussian_blur
jpeg_compression
```

### 4.1 Nessun attacco

Restituisce l'immagine stego invariata.

### 4.2 Rumore gaussiano

Parametri:

```text
sigma ∈ {5, 10, 20}
mean = 0
```

L'immagine deve essere convertita in float, perturbata, clippata e riconvertita in uint8.

### 4.3 Blur gaussiano

Parametri:

```text
kernel_size ∈ {3, 5, 7}
sigma opzionale
```

Il kernel deve essere dispari.

### 4.4 Compressione JPEG

Parametri:

```text
quality ∈ {90, 70, 50, 30}
```

Implementare salvando e ricaricando da buffer in memoria, non necessariamente da file fisico.

---

## 5. Metriche da calcolare

Tutte le metriche devono essere salvate in formato tabellare, idealmente CSV e JSONL.

### 5.1 Metriche di qualità visiva

Calcolare:

```text
PSNR_full
PSNR_roi
SSIM_full
SSIM_roi
SSIM_non_roi, opzionale
```

Interpretazione PSNR:

```text
> 40 dB: alterazione molto bassa
35-40 dB: buona qualità
30-35 dB: alterazione visibile ma spesso accettabile
< 30 dB: degrado evidente
```

Interpretazione SSIM:

```text
> 0.98: quasi identico
0.95-0.98: buona qualità
0.90-0.95: degradazione moderata
< 0.90: degradazione forte
```

Nota: `SSIM_roi` è molto importante perché l'embedding avviene nella ROI.

### 5.2 Metriche di messaggio

Calcolare:

```text
BER
MRR
exact_match
character_accuracy, se il payload è testuale
```

Definizioni:

```text
BER = bit_errati / bit_totali
MRR = messaggi_con_BER_0 / numero_messaggi
exact_match = True se l'intero messaggio ricostruito coincide
```

Per ogni singolo esperimento salvare:

```text
bit_errors
total_bits
BER
exact_match
```

L'MRR si calcola in aggregazione.

### 5.3 Metriche di capacità

Calcolare:

```text
payload_bits
bpp_roi
bpp_image
```

Definizioni:

```text
bpp_roi = payload_bits / pixel_roi
bpp_image = payload_bits / pixel_image
```

Queste metriche sono essenziali per confrontare ROI grandi, ROI piccole e immagine intera.

### 5.4 Metriche di robustezza

Per ogni attacco:

```text
BER_attack
ΔBER_attack = BER_attack - BER_no_attack
```

Il calcolo di `ΔBER_attack` può essere fatto in fase di aggregazione, usando il risultato `none` come baseline della stessa configurazione.

### 5.5 Metriche computazionali

Misurare:

```text
yolo_time_ms
embedding_time_ms
extraction_time_ms
svd_time_ms
attack_time_ms
total_time_ms
```

Dato che SVD è implementata da zero, salvare anche:

```text
svd_reconstruction_error = ||A - UΣVᵀ||_F / ||A||_F
```

---

## 6. Schema dei risultati

Ogni run deve produrre una riga con almeno questi campi:

```text
run_id
dataset
image_id
image_path
image_width
image_height

roi_strategy
roi_class_id
roi_class_name
roi_confidence
roi_x1
roi_y1
roi_x2
roi_y2
roi_width
roi_height
roi_area
roi_area_ratio
num_detections

svd_band
decoder_type
embedding_strength
payload_bits
payload_text
payload_seed
bpp_roi
bpp_image

attack_type
attack_strength
attack_param_sigma
attack_param_kernel
attack_param_quality

PSNR_full
PSNR_roi
SSIM_full
SSIM_roi

bit_errors
total_bits
BER
exact_match
character_accuracy

yolo_time_ms
embedding_time_ms
extraction_time_ms
svd_time_ms
attack_time_ms
total_time_ms
svd_reconstruction_error

status
error_message
```

`status` può assumere valori come:

```text
success
failed_no_detection
failed_payload_too_large
failed_svd
failed_decode
failed_unknown
```

---

## 7. Architettura proposta

Usare una struttura semplice e modulare.

```text
semantic_stego/
  __init__.py

  config/
    __init__.py
    defaults.py
    cli_args.py
    schemas.py

  data/
    __init__.py
    coco_loader.py
    image_io.py

  detection/
    __init__.py
    yolo_detector.py
    roi_selector.py

  svd/
    __init__.py
    svd_from_scratch.py
    svd_utils.py

  stego/
    __init__.py
    embedder.py
    extractor.py
    payload.py

  attacks/
    __init__.py
    attacks.py

  metrics/
    __init__.py
    image_metrics.py
    message_metrics.py
    timing.py

  experiments/
    __init__.py
    runner.py
    grid.py
    result_writer.py

  cli/
    __init__.py
    app.py

  main.py
tests/
  test_roi_selector.py
  test_payload.py
  test_svd.py
  test_embed_extract.py
  test_attacks.py
  test_metrics.py
scripts/
  run_debug.sh
  run_coco_ablation.sh
```

---

## 8. Responsabilità dei moduli

### 8.1 `config/defaults.py`

Contiene default globali, per esempio:

```python
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_ROI_STRATEGIES = ["largest", "smallest", "random", "full_image"]
DEFAULT_SVD_BANDS = ["high_energy", "mid_energy", "low_energy"]
DEFAULT_DECODERS = ["non_blind", "blind"]
DEFAULT_ATTACKS = ["none", "gaussian_noise", "gaussian_blur", "jpeg_compression"]
DEFAULT_JPEG_QUALITIES = [90, 70, 50, 30]
DEFAULT_NOISE_SIGMAS = [5, 10, 20]
DEFAULT_BLUR_KERNELS = [3, 5, 7]
```

### 8.2 `config/cli_args.py`

Definisce tutti gli argomenti CLI.

La logica dei parametri deve stare qui, non nei moduli principali.

Esempi di flag:

```bash
--coco-root data/coco
--split val2017
--output-dir outputs/debug
--max-images 50
--image-size 640
--yolo-model yolov8n.pt
--confidence-threshold 0.25
--roi-strategies largest smallest random full_image
--svd-bands high_energy mid_energy low_energy
--decoders non_blind blind
--attacks none gaussian_noise gaussian_blur jpeg
--jpeg-qualities 90 70 50 30
--noise-sigmas 5 10 20
--blur-kernels 3 5 7
--payload-text "secret message"
--payload-bits 128
--embedding-strength 10
--seed 42
--save-images
--save-roi-debug
```

### 8.3 `config/schemas.py`

Definire dataclass o Pydantic models, preferibilmente dataclass per semplicità.

Esempi:

```python
@dataclass
class ExperimentConfig:
    coco_root: Path
    split: str
    output_dir: Path
    max_images: int | None
    seed: int
    roi_strategies: list[str]
    svd_bands: list[str]
    decoders: list[str]
    attacks: list[str]
    embedding_strength: float
    payload_text: str | None
    payload_bits: int
```

```python
@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
```

```python
@dataclass
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    strategy: str
    class_id: int | None
    class_name: str | None
    confidence: float | None
```

### 8.4 `data/coco_loader.py`

Responsabilità:

- iterare sulle immagini COCO;
- restituire `image_id`, `image_path`;
- applicare `max_images`;
- supportare seed per campionamento riproducibile.

Non deve contenere logica YOLO, SVD o metriche.

### 8.5 `data/image_io.py`

Responsabilità:

- leggere immagini in RGB;
- salvare immagini;
- crop ROI;
- reinserire ROI modificata nell'immagine originale;
- conversioni `uint8`, `float32`.

### 8.6 `detection/yolo_detector.py`

Wrapper attorno a YOLO pretrained.

Responsabilità:

- caricare il modello;
- eseguire inference;
- restituire lista di `Detection`;
- filtrare per confidence;
- convertire coordinate a interi validi.

Il modulo deve nascondere dettagli della libreria YOLO usata.

Interfaccia consigliata:

```python
class YoloDetector:
    def __init__(self, model_name: str, confidence_threshold: float):
        ...

    def detect(self, image: np.ndarray) -> list[Detection]:
        ...
```

### 8.7 `detection/roi_selector.py`

Implementa:

```python
select_roi(
    image_shape: tuple[int, int, int],
    detections: list[Detection],
    strategy: str,
    rng: np.random.Generator,
) -> ROI | None
```

Regole:

- `largest`: massimo `area`;
- `smallest`: minimo `area`;
- `random`: scelta casuale riproducibile;
- `full_image`: ROI intera;
- se nessuna detection e strategia non `full_image`, restituisce `None`.

### 8.8 `svd/svd_from_scratch.py`

Implementare SVD da zero in Python/Numpy.

Nota importante: non usare `np.linalg.svd` nella pipeline finale. Può essere usato solo nei test come riferimento, se necessario.

Approccio pratico consigliato:

Per una matrice `A`:

1. calcolare `A.T @ A`;
2. calcolare autovalori/autovettori con algoritmo implementato oppure, se il requisito "da zero" consente solo di evitare SVD builtin, usare `np.linalg.eigh` come compromesso;
3. valori singolari `S = sqrt(eigenvalues)`;
4. `V = eigenvectors`;
5. `U = A @ V / S`;
6. gestire valori singolari prossimi a zero.

Opzione più rigorosa:

- implementare power iteration / deflation per autovalori;
- o Jacobi eigenvalue algorithm per matrici simmetriche.

Decisione suggerita per il coding agent:

```text
Implementare una SVD didattica autonoma basata su eigendecomposition simmetrica.
Evitare np.linalg.svd.
Documentare chiaramente se np.linalg.eigh viene usato come backend ausiliario.
```

Interfaccia:

```python
def svd_decompose(A: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

def svd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    ...
```

### 8.9 `svd/svd_utils.py`

Funzioni:

```python
select_singular_indices(S: np.ndarray, payload_len: int, band: str) -> np.ndarray
compute_reconstruction_error(A, U, S, Vt) -> float
```

### 8.10 `stego/payload.py`

Responsabilità:

- convertire testo in bit;
- convertire bit in testo;
- generare payload casuale di lunghezza fissa;
- padding/truncation opzionali.

Funzioni consigliate:

```python
text_to_bits(text: str, encoding: str = "utf-8") -> np.ndarray
bits_to_text(bits: np.ndarray, encoding: str = "utf-8") -> str
random_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray
```

Per esperimenti riproducibili, preferire payload random a lunghezza fissa, salvando seed e bit length.

### 8.11 `stego/embedder.py`

Responsabilità:

- ricevere una ROI;
- convertire eventualmente in canale/luminanza;
- calcolare SVD;
- modificare valori singolari;
- ricostruire ROI stego;
- reinserire ROI nell'immagine.

Opzioni colore:

Semplice default consigliato:

```text
Convertire ROI RGB in YCbCr e applicare embedding solo sul canale Y.
```

Alternativa più semplice:

```text
Applicare embedding sul canale grayscale derivato dalla ROI e poi reinserire sul canale luminanza.
```

Evitare embedding indipendente su tutti e tre i canali all'inizio, perché aumenta complessità e artefatti.

Interfaccia:

```python
class SvdEmbedder:
    def embed(
        self,
        image: np.ndarray,
        roi: ROI,
        payload_bits: np.ndarray,
        band: str,
        strength: float,
        mode: str,
    ) -> EmbeddingResult:
        ...
```

`EmbeddingResult` deve contenere:

```text
stego_image
metadata
svd_time_ms
embedding_time_ms
svd_reconstruction_error
```

Metadata minimi:

```text
roi
band
indices
payload_len
strength
mode
```

Per `blind`, i metadata non devono includere l'immagine originale o i valori singolari originali. Possono includere lunghezza payload, band e strength.

### 8.12 `stego/extractor.py`

Responsabilità:

- estrarre il payload dalla stego image;
- supportare `non_blind` e `blind`.

Interfaccia:

```python
class SvdExtractor:
    def extract(
        self,
        stego_or_attacked_image: np.ndarray,
        metadata: EmbeddingMetadata,
        original_image: np.ndarray | None,
        decoder_type: str,
    ) -> ExtractionResult:
        ...
```

Per `non_blind`:

- usare `original_image`;
- calcolare SVD originale e stego sulla stessa ROI;
- dedurre bit in base alla differenza o alla regola usata in embedding.

Per `blind`:

- non accedere a `original_image`;
- usare QIM o regola equivalente;
- ricostruire bit dai valori singolari quantizzati.

### 8.13 `attacks/attacks.py`

Interfaccia uniforme:

```python
def apply_attack(image: np.ndarray, attack_type: str, params: dict) -> np.ndarray:
    ...
```

Implementare:

```python
apply_gaussian_noise(image, sigma, mean=0)
apply_gaussian_blur(image, kernel_size, sigma=None)
apply_jpeg_compression(image, quality)
```

### 8.14 `metrics/image_metrics.py`

Implementare:

```python
compute_psnr(original, modified) -> float
compute_ssim(original, modified) -> float
compute_roi_metrics(original, modified, roi) -> dict
```

È accettabile usare `skimage.metrics` per PSNR/SSIM.

### 8.15 `metrics/message_metrics.py`

Implementare:

```python
bit_error_rate(original_bits, recovered_bits) -> float
bit_errors(original_bits, recovered_bits) -> int
exact_match(original_bits, recovered_bits) -> bool
character_accuracy(original_text, recovered_text) -> float
```

Gestire lunghezze diverse.

### 8.16 `metrics/timing.py`

Context manager semplice:

```python
with Timer() as t:
    ...
elapsed_ms = t.elapsed_ms
```

### 8.17 `experiments/grid.py`

Genera combinazioni sperimentali.

Configurazione di default:

```text
roi_strategy ∈ {largest, smallest, random, full_image}
svd_band ∈ {high_energy, mid_energy, low_energy}
decoder_type ∈ {non_blind, blind}
attack_type ∈ {
  none,
  gaussian_noise sigma 5/10/20,
  gaussian_blur kernel 3/5/7,
  jpeg quality 90/70/50/30
}
```

Numero configurazioni per immagine:

```text
4 ROI strategies
× 3 SVD bands
× 2 decoders
× 11 attack conditions
= 264 configurazioni per immagine
```

Dove gli 11 attacchi sono:

```text
none = 1
gaussian_noise = 3
gaussian_blur = 3
jpeg = 4
totale = 11
```

Per una versione più leggera di debug:

```text
none
gaussian_noise sigma=5
gaussian_blur kernel=3
jpeg quality=90
```

Numero configurazioni debug:

```text
4 × 3 × 2 × 4 = 96 configurazioni per immagine
```

### 8.18 `experiments/runner.py`

Responsabilità:

1. carica immagini COCO;
2. esegue YOLO una volta per immagine;
3. per ogni strategia ROI seleziona ROI;
4. per ogni band SVD esegue embedding;
5. per ogni decoder e attacco esegue attacco + extraction;
6. calcola metriche;
7. scrive risultati incrementalmente.

Importante: YOLO va eseguito una sola volta per immagine, non per configurazione.

Pseudo-pipeline:

```python
for image_record in coco_loader:
    image = read_image(image_record.path)
    detections = detector.detect(image)

    for roi_strategy in config.roi_strategies:
        roi = select_roi(image.shape, detections, roi_strategy, rng)

        if roi is None:
            write_failed_result(...)
            continue

        for svd_band in config.svd_bands:
            payload_bits = get_payload(config, roi)

            embed_result = embedder.embed(
                image=image,
                roi=roi,
                payload_bits=payload_bits,
                band=svd_band,
                strength=config.embedding_strength,
                mode="qim_or_delta"
            )

            clean_stego = embed_result.stego_image

            for attack_config in attack_grid:
                attacked = apply_attack(clean_stego, attack_config.type, attack_config.params)

                for decoder_type in config.decoders:
                    original_for_decoder = image if decoder_type == "non_blind" else None

                    extract_result = extractor.extract(
                        attacked,
                        metadata=embed_result.metadata,
                        original_image=original_for_decoder,
                        decoder_type=decoder_type
                    )

                    metrics = compute_all_metrics(
                        original=image,
                        stego_or_attacked=attacked,
                        roi=roi,
                        payload_bits=payload_bits,
                        recovered_bits=extract_result.bits
                    )

                    write_result(...)
```

### 8.19 `experiments/result_writer.py`

Scrivere risultati in:

```text
outputs/<run_name>/
  config.json
  results.csv
  results.jsonl
  failures.jsonl
  images/
    optional
  roi_debug/
    optional
```

Il writer deve fare flush periodico per non perdere risultati su run lunghi.

---

## 9. CLI

Implementare comando principale:

```bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/coco_debug \
  --max-images 50 \
  --yolo-model yolov8n.pt \
  --confidence-threshold 0.25 \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg \
  --noise-sigmas 5 \
  --blur-kernels 3 \
  --jpeg-qualities 90 \
  --payload-bits 128 \
  --embedding-strength 10 \
  --seed 42 \
  --save-roi-debug
```

La CLI deve:

1. parsare argomenti;
2. costruire `ExperimentConfig`;
3. salvare `config.json`;
4. chiamare `ExperimentRunner`;
5. non contenere logica di embedding/detection/metriche.

---

## 10. Main semplice

Implementare anche `main.py` come script minimale:

```python
from semantic_stego.config.defaults import build_default_debug_config
from semantic_stego.experiments.runner import ExperimentRunner

def main():
    config = build_default_debug_config()
    runner = ExperimentRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
```

Questo deve permettere esecuzione semplice senza CLI complessa.

---

## 11. Esperimenti consigliati

### 11.1 Fase 1: debug funzionale

Dataset:

```text
COCO val2017, 50 immagini
```

Configurazione:

```text
roi_strategy: largest, smallest, random, full_image
svd_band: high_energy, mid_energy, low_energy
decoder: non_blind, blind
attacks: none, gaussian_noise sigma=5, gaussian_blur kernel=3, jpeg quality=90
payload_bits: 128
embedding_strength: 10
```

Obiettivo:

- pipeline end-to-end funzionante;
- risultati CSV corretti;
- salvataggio errori;
- verifica BER e qualità immagine.

### 11.2 Fase 2: ablation principale

Dataset:

```text
COCO val2017, 500 immagini filtrate
```

Configurazione completa:

```text
roi_strategy: largest, smallest, random, full_image
svd_band: high_energy, mid_energy, low_energy
decoder: non_blind, blind
attacks:
  none
  gaussian_noise sigma 5/10/20
  gaussian_blur kernel 3/5/7
  jpeg quality 90/70/50/30
payload_bits: 128 oppure 256
embedding_strength: testare 5, 10, 20
```

Nota: se si testano anche più valori di `embedding_strength`, la griglia cresce molto. Implementare strength come parametro singolo nella prima versione e come lista opzionale nella seconda.

### 11.3 Fase 3: esperimenti su capacità

Testare payload diversi:

```text
payload_bits ∈ {64, 128, 256, 512}
```

Obiettivo:

- capire il trade-off tra capacità, qualità e BER;
- valutare il comportamento su ROI piccole.

---

## 12. Analisi finale attesa

Il sistema deve permettere aggregazioni come:

### 12.1 Confronto ROI

```text
groupby roi_strategy:
  mean BER
  mean PSNR_roi
  mean SSIM_roi
  MRR
  mean bpp_roi
```

### 12.2 Confronto bande SVD

```text
groupby svd_band:
  mean BER
  mean PSNR_roi
  mean SSIM_roi
  MRR
```

Ipotesi attesa:

- `high_energy`: più robusta, ma più visibile;
- `mid_energy`: possibile compromesso;
- `low_energy`: meno visibile, ma più fragile.

### 12.3 Confronto decoder

```text
groupby decoder_type:
  mean BER
  MRR
```

Ipotesi attesa:

- `non_blind` dovrebbe performare meglio;
- `blind` è più realistico ma più difficile.

### 12.4 Confronto attacchi

```text
groupby attack_type, attack_strength:
  mean BER
  MRR
  delta BER rispetto a none
```

Ipotesi attesa:

- JPEG a qualità bassa dovrebbe essere uno degli attacchi più distruttivi;
- blur può distruggere componenti ad alta frequenza;
- rumore gaussiano peggiora con sigma crescente.

### 12.5 Confronto per classe YOLO

```text
groupby roi_class_name:
  mean BER
  mean PSNR_roi
  mean roi_area_ratio
```

Utile per capire se alcune classi sono ROI migliori.

---

## 13. Criteri di accettazione

La prima versione è accettabile se:

1. legge immagini COCO da `val2017`;
2. esegue YOLO pretrained;
3. seleziona ROI con `largest`, `smallest`, `random`, `full_image`;
4. implementa SVD senza usare `np.linalg.svd` nella pipeline;
5. embeddizza e ricostruisce una ROI;
6. supporta almeno payload binario fisso;
7. supporta decoding `non_blind`;
8. implementa una baseline `blind`;
9. applica i tre attacchi richiesti;
10. calcola almeno:
    - PSNR full;
    - PSNR ROI;
    - SSIM full;
    - SSIM ROI;
    - BER;
    - exact match;
    - bpp ROI;
    - bpp image;
11. salva risultati in CSV;
12. salva configurazione in JSON;
13. ha test unitari per ROI selector, payload, SVD, embedding/extraction, attacchi e metriche.

---

## 14. Test unitari minimi

### 14.1 ROI selector

Testare:

- `largest` seleziona box con area massima;
- `smallest` seleziona box con area minima;
- `random` è riproducibile con seed;
- `full_image` restituisce coordinate immagine intera;
- nessuna detection restituisce `None` per strategie YOLO.

### 14.2 Payload

Testare:

- `text_to_bits` e `bits_to_text`;
- payload casuale riproducibile;
- gestione padding/truncation.

### 14.3 SVD

Testare:

- ricostruzione approssimata di matrice piccola;
- valori singolari non negativi;
- errore di ricostruzione sotto soglia;
- confronto opzionale con `np.linalg.svd` solo nei test.

### 14.4 Embedding/extraction

Testare:

- embedding modifica immagine;
- dimensioni immagine invariate;
- extraction `non_blind` recupera payload senza attacco;
- extraction `blind` recupera payload almeno in condizioni clean per payload piccolo e strength adeguata.

### 14.5 Attacchi

Testare:

- output stessa shape;
- output `uint8`;
- JPEG cambia immagine con qualità bassa;
- blur accetta solo kernel dispari;
- noise rispetta clipping.

### 14.6 Metriche

Testare:

- PSNR infinito o molto alto su immagini identiche;
- BER = 0 per bit identici;
- BER corretto per array con errori noti;
- exact match corretto.

---

## 15. Note implementative importanti

### 15.1 Gestione colore

Consiglio:

1. leggere immagine in RGB;
2. convertire ROI in YCbCr;
3. applicare SVD sul canale Y;
4. ricostruire Y modificato;
5. riunire Y modificato con CbCr originali;
6. riconvertire in RGB;
7. reinserire ROI nell'immagine.

Questa scelta mantiene l'embedding più controllato.

### 15.2 Coordinate ROI

Tutte le coordinate devono essere clippate:

```text
x1 >= 0
y1 >= 0
x2 <= image_width
y2 <= image_height
x2 > x1
y2 > y1
```

### 15.4 Riproducibilità

Usare un seed unico:

```text
seed = 42
```

Derivare RNG da questo seed per:

- selezione immagini;
- ROI random;
- payload random.

Salvare sempre il seed nel `config.json`.

### 15.5 Salvataggio immagini

Per default, non salvare tutte le immagini stego per evitare output enorme.

Flag opzionali:

```text
--save-images
--save-roi-debug
```

Salvare immagini solo in debug o per subset.

### 15.6 Logging

Usare logging standard Python.

Livelli:

```text
INFO: progresso generale
WARNING: immagini saltate
ERROR: errori non bloccanti su singole configurazioni
DEBUG: dettagli opzionali
```

### 15.7 Error handling

Il runner non deve fermarsi su una singola immagine/configurazione fallita.

Ogni errore deve produrre una riga `failed_*` con messaggio.

---

## 16. Dipendenze consigliate

Minime:

```text
numpy
opencv-python
Pillow
scikit-image
pandas
tqdm
ultralytics
pytest
```

Opzionali:

```text
pydantic
rich
```

Evitare dipendenze non necessarie.

---

## 17. Output finale atteso

Esempio struttura:

```text
outputs/coco_ablation_001/
  config.json
  results.csv
  results.jsonl
  failures.jsonl
  summary.csv
  README_RUN.md
  roi_debug/
    000000000139_largest.jpg
    ...
```

`summary.csv` può essere prodotto da uno script di aggregazione successivo, ma è utile includerlo.

Aggregazioni minime:

```text
mean/std BER by roi_strategy
mean/std BER by svd_band
mean/std BER by decoder_type
mean/std BER by attack_type
mean/std PSNR_roi by configuration
MRR by configuration
```

---

## 18. Roadmap implementativa consigliata

### Step 1: Setup progetto

- creare struttura cartelle;
- configurare ambiente;
- aggiungere `pyproject.toml` o `requirements.txt`;
- aggiungere test skeleton.

### Step 2: Data loader COCO

- iterazione immagini;
- supporto `max_images`;
- lettura immagini RGB.

### Step 3: YOLO wrapper

- caricare modello pretrained;
- detection su una immagine;
- conversione output in dataclass `Detection`.

### Step 4: ROI selector

- implementare le quattro strategie;
- test unitari completi.

### Step 5: Payload

- bit random;
- testo ↔ bit;
- test unitari.

### Step 6: SVD from scratch

- implementare decomposizione;
- implementare ricostruzione;
- test su matrici piccole;
- calcolare reconstruction error.

### Step 7: Embedding SVD

- embedding su canale Y;
- selezione banda;
- reinserimento ROI;
- salvataggio metadata.

### Step 8: Extraction

- implementare `non_blind`;
- implementare baseline `blind` con QIM;
- test clean senza attacchi.

### Step 9: Attacchi

- noise;
- blur;
- JPEG;
- test unitari.

### Step 10: Metriche

- PSNR/SSIM full;
- PSNR/SSIM ROI;
- BER/exact match;
- bpp.

### Step 11: Runner esperimenti

- loop immagini;
- loop configurazioni;
- writer incrementale;
- gestione errori.

### Step 12: CLI

- argparse;
- config;
- comando end-to-end.

### Step 13: Main semplice

- default debug config;
- esecuzione rapida.

### Step 14: Debug run

Eseguire:

```bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/debug \
  --max-images 10 \
  --roi-strategies largest full_image \
  --svd-bands mid_energy \
  --decoders non_blind \
  --attacks none \
  --payload-bits 64 \
  --embedding-strength 10 \
  --seed 42 \
  --save-roi-debug
```

### Step 15: Run ablation ridotta

Eseguire:

```bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/coco_ablation_small \
  --max-images 50 \
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

---

## 19. Priorità tecniche

Ordine di priorità:

1. pipeline end-to-end funzionante;
2. risultati riproducibili;
3. metriche corrette;
4. test unitari;
5. ottimizzazioni;
6. supporto esperimenti grandi;
7. miglioramento blind decoding.

Non ottimizzare prematuramente SVD o YOLO. Prima produrre dati corretti e verificabili.

---

## 20. Rischi principali

### Rischio 1: blind decoding fragile

Mitigazione:

- implementare QIM;
- aumentare `embedding_strength`;
- testare payload corto;
- confrontare sempre con `non_blind`.

### Rischio 2: ROI piccole non supportano payload

Mitigazione:

- calcolare capacità prima dell'embedding;
- salvare failure;
- usare `payload_bits` moderato, tipo 64 o 128.

### Rischio 3: alterazione visiva troppo alta

Mitigazione:

- testare strength diversi;
- confrontare PSNR/SSIM;
- preferire mid/low energy se high energy è troppo visibile.

### Rischio 4: SVD da zero lenta

Mitigazione:

- limitare dimensione ROI in debug;
- opzionalmente resize ROI;
- misurare `svd_time_ms`;
- mantenere backend SVD sostituibile.

### Rischio 5: griglia sperimentale troppo grande

Mitigazione:

- usare `max_images`;
- usare debug grid;
- salvare risultati incrementalmente;
- supportare resume in una versione successiva.

---

## 21. Decisione tecnica consigliata per la prima versione

Per massimizzare probabilità di successo:

```text
Dataset: COCO val2017
YOLO: yolov8n.pt o altro pretrained leggero
Payload: random bits, 128 bit
Embedding: canale Y in YCbCr
Blind baseline: QIM sui valori singolari
Non-blind: confronto valori singolari originali/stego
ROI: largest, smallest, random, full_image
SVD band: high_energy, mid_energy, low_energy
Attacchi debug: none, noise sigma 5, blur kernel 3, JPEG quality 90
Metriche: PSNR, SSIM, BER, exact_match, bpp_roi, bpp_image, tempi
```

Questa configurazione è sufficiente per realizzare una prima pipeline solida, testabile e poi estendibile.

