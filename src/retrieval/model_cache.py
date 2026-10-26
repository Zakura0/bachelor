"""
Prozessweiter Cache für Modelle, damit jedes Modell nur einmal im (GPU-)Speicher liegt.
"""
import threading
from typing import Any, Callable

_models: dict[tuple, tuple[Any, threading.Lock]] = {}
_load_lock = threading.Lock()


def get_shared_model(key: tuple, factory: Callable[[], Any]) -> tuple[Any, threading.Lock]:
    """
    Liefert das geteilte Modell zu ``key`` und lädt es beim ersten Zugriff über ``factory``.

    :param key: Eindeutiger Schlüssel, z.B. (Modelltyp, Modellname, Device)
    :param factory: Funktion, die das Modell erzeugt
    :return: Tupel aus Modell und Lock, der um jede Inferenz gehalten werden muss
        (Tokenizer und GPU-Inferenz sind nicht threadsicher)
    """
    with _load_lock:
        if key not in _models:
            _models[key] = (factory(), threading.Lock())
        return _models[key]
