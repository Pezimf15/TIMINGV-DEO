# Ferramenta de análise de locuções de varejo

Este repositório contém um script Python que transcreve locuções de vídeos de varejo,
identifica introdução, produtos e fechamento e calcula a duração de cada trecho.

## Dependências

Instale os requisitos com:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Os modelos Whisper são baixados automaticamente na primeira execução.

## Uso

```bash
python analisar_audio_varejo.py --audio caminho/para/audio.mp3 --saida resultado.json
```

Parâmetros principais:

- `--audio`: caminho do arquivo de áudio a ser analisado.
- `--saida`/`--output`: caminho do arquivo de saída (`.json` ou `.csv`).
- `--modelo`: modelo Whisper a ser utilizado (padrão `small`).
- `--device`: `cpu` ou `cuda`.
- `--verbose`: habilita logs detalhados.

## Exemplo de saída JSON

```json
{
  "file_name": "esquenta_black_chame_chame.mp3",
  "total_duration_sec": 29.5,
  "intro": {
    "start_time_sec": 0.0,
    "end_time_sec": 3.2,
    "duration_sec": 3.2
  },
  "products": [
    {
      "product_text": "MASSA CORRIDA CORAL 69,90",
      "start_time_sec": 3.2,
      "end_time_sec": 5.8,
      "duration_sec": 2.6
    },
    {
      "product_text": "TINTA INTEROR GLASU 18L 209,00",
      "start_time_sec": 5.8,
      "end_time_sec": 8.9,
      "duration_sec": 3.1
    }
  ],
  "closing": {
    "start_time_sec": 24.0,
    "end_time_sec": 29.5,
    "duration_sec": 5.5
  }
}
```

O resumo também é impresso no terminal com o tempo de cada segmento identificado.
