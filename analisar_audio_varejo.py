"""Ferramenta para analisar locuções de vídeos de varejo.

O script transcreve um arquivo de áudio utilizando Whisper, identifica
introdução, produtos e fechamento e exporta um resumo com os tempos.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import whisper


LOGGER = logging.getLogger(__name__)

# Expressões regulares para detectar preços no padrão brasileiro.
PRICE_REGEXES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b"),  # 1.234,56
    re.compile(r"\b\d+[,.]\d{2}\b"),  # 12,34 ou 12.34
    re.compile(r"\b\d{2,4}\b"),  # fallback para valores falados sem separador
)

# Palavras comuns em descrições de produtos de materiais de construção.
PRODUCT_KEYWORDS = {
    "tinta",
    "massa",
    "kit",
    "bloco",
    "pisos",
    "piso",
    "balde",
    "litro",
    "litros",
    "l",
    "ml",
    "kg",
    "parafuso",
    "pincel",
    "rolo",
    "condor",
    "tigre",
    "coral",
    "tropical",
    "vidro",
    "caixa",
}


@dataclass
class Segment:
    """Representa um trecho da transcrição."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_arguments() -> argparse.Namespace:
    """Configura e interpreta os argumentos da linha de comando."""

    parser = argparse.ArgumentParser(description="Analisa uma locução de vídeo de varejo")
    parser.add_argument("--audio", required=True, help="Caminho para o arquivo de áudio (mp3, wav etc.)")
    parser.add_argument("--saida", "--output", dest="output", required=True, help="Caminho do arquivo de saída (.json ou .csv)")
    parser.add_argument(
        "--modelo",
        "--model",
        dest="model_name",
        default="small",
        help="Nome do modelo Whisper a ser utilizado (padrão: small)",
    )
    parser.add_argument(
        "--idioma",
        "--language",
        dest="language",
        default="pt",
        help="Idioma da transcrição (padrão: pt)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para executar o Whisper (cpu ou cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe logs detalhados durante a execução",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Inicializa o logging do módulo."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")


def load_model(model_name: str, device: str) -> whisper.Whisper:
    """Carrega o modelo Whisper especificado."""

    LOGGER.info("Carregando modelo Whisper '%s' no dispositivo %s...", model_name, device)
    return whisper.load_model(model_name, device=device)


def transcribe_audio(model: whisper.Whisper, audio_path: Path, language: str) -> List[Segment]:
    """Executa a transcrição do áudio e retorna os segmentos com timestamps."""

    LOGGER.info("Iniciando transcrição de %s", audio_path)
    result = model.transcribe(str(audio_path), language=language)
    segments: List[Segment] = []
    for seg in result.get("segments", []):
        segments.append(Segment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip()))
    LOGGER.info("Transcrição concluída com %d segmentos", len(segments))
    return segments


def contains_price(text: str) -> bool:
    """Retorna True se o texto contiver algo que pareça um preço."""

    normalized = text.replace("R$", "").replace("reais", "").replace("RS", "").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    for regex in PRICE_REGEXES:
        if regex.search(normalized):
            return True
    return False


def is_product_candidate(text: str) -> bool:
    """Verifica se um texto se parece com a descrição de um produto."""

    lower = text.lower()
    if contains_price(lower):
        return True
    if re.search(r"\b\d+[lkgm]?\b", lower):
        return True
    if any(keyword in lower for keyword in PRODUCT_KEYWORDS):
        return True
    return False


def aggregate_segments(segments: Sequence[Segment]) -> dict:
    """Classifica e agrega os segmentos por introdução, produtos e fechamento."""

    if not segments:
        raise ValueError("Nenhum segmento retornado pela transcrição")

    total_duration = segments[-1].end if segments else 0.0

    # Identifica blocos de produtos percorrendo os segmentos em sequência.
    product_ranges: List[tuple[int, int]] = []
    current_start: Optional[int] = None
    first_product_idx: Optional[int] = None

    for idx, segment in enumerate(segments):
        text = segment.text
        if current_start is None and is_product_candidate(text):
            current_start = idx
            if first_product_idx is None:
                first_product_idx = idx
        if current_start is not None and contains_price(text):
            product_ranges.append((current_start, idx))
            current_start = None

    # Caso o áudio termine sem repetir o preço (por erro de transcrição),
    # consideramos qualquer bloco em aberto até o final como produto.
    if current_start is not None:
        product_ranges.append((current_start, len(segments) - 1))
        if first_product_idx is None:
            first_product_idx = current_start

    if not product_ranges:
        intro_segment = Segment(start=segments[0].start, end=segments[-1].end, text=" ".join(seg.text for seg in segments))
        return {
            "total_duration_sec": total_duration,
            "intro": {
                "start_time_sec": intro_segment.start,
                "end_time_sec": intro_segment.end,
                "duration_sec": intro_segment.duration,
            },
            "products": [],
            "closing": None,
        }

    intro_end_idx = product_ranges[0][0] - 1
    if intro_end_idx >= 0:
        intro_segment = Segment(start=segments[0].start, end=segments[intro_end_idx].end, text=" ".join(seg.text for seg in segments[: product_ranges[0][0]]))
    else:
        intro_segment = None

    closing_start_idx = product_ranges[-1][1] + 1
    if closing_start_idx < len(segments):
        closing_segment = Segment(start=segments[closing_start_idx].start, end=segments[-1].end, text=" ".join(seg.text for seg in segments[closing_start_idx:]))
    else:
        closing_segment = None

    product_entries = []
    for start_idx, end_idx in product_ranges:
        merged_text = " ".join(seg.text for seg in segments[start_idx : end_idx + 1]).strip()
        product_segment = Segment(start=segments[start_idx].start, end=segments[end_idx].end, text=merged_text)
        product_entries.append(
            {
                "product_text": product_segment.text,
                "start_time_sec": product_segment.start,
                "end_time_sec": product_segment.end,
                "duration_sec": product_segment.duration,
            }
        )

    result = {
        "total_duration_sec": total_duration,
        "intro": None,
        "products": product_entries,
        "closing": None,
    }

    if intro_segment is not None:
        result["intro"] = {
            "start_time_sec": intro_segment.start,
            "end_time_sec": intro_segment.end,
            "duration_sec": intro_segment.duration,
        }

    if closing_segment is not None:
        result["closing"] = {
            "start_time_sec": closing_segment.start,
            "end_time_sec": closing_segment.end,
            "duration_sec": closing_segment.duration,
        }

    return result


def export_json(output_path: Path, audio_path: Path, summary: dict) -> None:
    """Gera um arquivo JSON com o resumo da análise."""

    payload = {"file_name": audio_path.name}
    payload.update(summary)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    LOGGER.info("Arquivo JSON gerado em %s", output_path)


def export_csv(output_path: Path, audio_path: Path, summary: dict) -> None:
    """Gera um CSV com as durações dos segmentos."""

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", audio_path.name])
        if summary.get("intro"):
            intro = summary["intro"]
            writer.writerow(["intro", intro["start_time_sec"], intro["end_time_sec"], intro["duration_sec"]])
        for idx, product in enumerate(summary.get("products", []), start=1):
            writer.writerow([
                f"product_{idx}",
                product["product_text"],
                product["start_time_sec"],
                product["end_time_sec"],
                product["duration_sec"],
            ])
        if summary.get("closing"):
            closing = summary["closing"]
            writer.writerow(["closing", closing["start_time_sec"], closing["end_time_sec"], closing["duration_sec"]])
    LOGGER.info("Arquivo CSV gerado em %s", output_path)


def print_summary(summary: dict) -> None:
    """Exibe um resumo amigável no terminal."""

    intro = summary.get("intro")
    if intro:
        print(f"Introdução: {intro['duration_sec']:.2f}s")
    else:
        print("Introdução: não identificada")

    for idx, product in enumerate(summary.get("products", []), start=1):
        print(
            f"Produto {idx} ({product['product_text']}): {product['duration_sec']:.2f}s"
        )

    closing = summary.get("closing")
    if closing:
        print(f"Fechamento: {closing['duration_sec']:.2f}s")
    else:
        print("Fechamento: não identificado")

    print(f"Duração total: {summary.get('total_duration_sec', 0.0):.2f}s")


def main() -> None:
    args = parse_arguments()
    configure_logging(args.verbose)

    audio_path = Path(args.audio)
    output_path = Path(args.output)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    model = load_model(args.model_name, args.device)
    segments = transcribe_audio(model, audio_path, args.language)
    summary = aggregate_segments(segments)

    # Ajusta o formato de saída com base na extensão.
    if output_path.suffix.lower() == ".json":
        export_json(output_path, audio_path, summary)
    elif output_path.suffix.lower() == ".csv":
        export_csv(output_path, audio_path, summary)
    else:
        raise ValueError("A extensão do arquivo de saída deve ser .json ou .csv")

    print_summary(summary)


if __name__ == "__main__":
    main()
