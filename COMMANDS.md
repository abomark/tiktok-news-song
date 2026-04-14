# CurrentNoise — kommandoreferanse

## Streamlit-dashboard

```bash
python -m streamlit run app.py
```

Åpner dashbordet på http://localhost:8501

---

## Pipeline

### Full kjøring (nyheter → tekst → musikk → video → TikTok-publisering)

```bash
python pipeline.py
```

### Tørrkjøring — alt unntatt publisering til TikTok

```bash
python pipeline.py --dry-run
```

### Tørrkjøring — kun nyheter + tekst (hopper over Suno, video, TikTok)

```bash
python pipeline.py --dry-run-full
```

### Kun generer tekst/lyrics, stopp der

```bash
python pipeline.py --lyrics-only
```

### Bruk Grok i stedet for Ollama for lyrics

```bash
python pipeline.py --provider grok
```

### Bruk egendefinert overskrift og sammendrag

```bash
python pipeline.py --headline "Custom headline" --summary "Custom summary"
```

### Kombiner flagg

```bash
python pipeline.py --dry-run --provider grok --headline "Headline" --summary "Summary"
```

---

## Delmoduler (kjør enkelttrinn)

### Lyrics-generator

```bash
python -m modules.lyrics_generator
python -m modules.lyrics_generator --provider grok
python -m modules.lyrics_generator --headline "Headline" --summary "Summary"
python -m modules.lyrics_generator --provider ollama --model gemma3 --date 2026-04-13 --run 01
```

### Musikk-generator (Suno)

```bash
python -m modules.music_generator
```

### Klipp-generator (Runway ML)

```bash
# Generer nye klipp for siste kjøring
python -m modules.clip_generator --reuse

# For spesifikk dato og run-mappe
python -m modules.clip_generator --reuse --date 2026-04-13 --run 01
python -m modules.clip_generator --reuse --date 2026-04-13 --run tariff-time
```

### Video-montering + teksting (full videogenerering)

```bash
# Full videogenerering fra eksisterende headlines.txt + lyrics.txt
python -m modules.video_generator --reuse

# Hopp over klippgenerering, bruk eksisterende klipp
python -m modules.video_generator --reuse --skip-clips

# Hopp over montering, bruk eksisterende final.mp4
python -m modules.video_generator --reuse --skip-assemble

# Hopp over teksting
python -m modules.video_generator --reuse --skip-captions

# Velg Whisper-modell (tiny/base/small/medium/large)
python -m modules.video_generator --reuse --whisper-model small

# Spesifikk dato og run
python -m modules.video_generator --reuse --date 2026-04-13 --run 01

# Sett lydsporvarighet manuelt
python -m modules.video_generator --reuse --duration 45.0
```

### Teksting alene (captioner)

```bash
python -m modules.captioner
python -m modules.captioner --date 2026-04-13 --run 01
python -m modules.captioner --whisper-model small
python -m modules.captioner --input output/2026-04-13/01-tariff-time/final.mp4 \
                             --audio  output/2026-04-13/01-tariff-time/song.mp3 \
                             --lyrics output/2026-04-13/01-tariff-time/lyrics.txt \
                             --output output/2026-04-13/01-tariff-time/final_captioned.mp4
```

---

## Planlegger (kjør pipeline automatisk kl. 09:00 hver dag)

```bash
python scheduler.py
```

---

## TikTok OAuth-oppsett

```bash
python tiktok_oauth_setup.py
```

---

## Installasjon og oppsett

### Installer avhengigheter

```bash
pip install -r requirements.txt
```

### Kopier og fyll ut miljøvariabler

```bash
cp .env.example .env
# Rediger .env med dine API-nøkler
```

---

## Output-mappestruktur

```
output/
└── 2026-04-13/
    └── 01-tariff-time/
        ├── headline.txt
        ├── lyrics.txt
        ├── song.mp3
        ├── clip_00.mp4
        ├── clip_01.mp4
        ├── final.mp4
        ├── final_captioned.mp4
        └── timed_lyrics.json
logs/
└── api_calls.jsonl
```
