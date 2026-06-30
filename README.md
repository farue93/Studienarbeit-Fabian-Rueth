# Studienarbeit Fabian Rüth

Bildbasierte Eisdickenmessung an Profilen im Windkanal — Vorverarbeitung der
Laserlinie. TU Braunschweig.

## Inhalt

| Pfad | Beschreibung |
|------|--------------|
| `raw/` | Roh-TIFs der Baumer-VCXU.2-241M-Kamera (Referenzframes, Frame 0) |
| `output/` | Zielordner für die erzeugten Ergebnisse (leer im Repo) |
| `laser_pipeline.py` | Vollständige Vorverarbeitungs-Pipeline (Detektion + Fit + ROI) |

## `laser_pipeline.py`

Erkennt die Laserlinie im eisfreien Referenzframe, fittet sie als
subpixelgenaue 1px-Linie und extrahiert die Mess-Grundlage. Ablauf:

1. Laden + Graustufen
2. Multi-Scale-Steger-Ridge-Detektion → alle Linienpixel + Stärke |λ_min|
3. Otsu-Kern + 2D-Dilation + Perzentilfilter („Steger4") → saubere Pixelmenge
4. Skelettierung + Pfadverfolgung → geordnete Teilstücke
5. Stärke-gewichteter Querschnitts-Schwerpunkt + Spline je Teilstück
6. Hermite-Brücke zwischen Teilstücken + lineare Endverlängerung
7. Ausgabe: 1px-Linie, Bogenlänge s, Normalen, ROI-Crop

Die senkrechte Verschiebung dieser Linie (Eis gegenüber Referenz) über der
Bogenlänge s ergibt später die Eisdicke; die Normalen geben die Messrichtung.

## Ausführen

```bash
python laser_pipeline.py
```

Verarbeitet alle `*.tif` in `raw/` und schreibt je Bild nach `output/`:
`*_laserlinie.npz` (x, y, s, nx, ny, …), `*_fit.png`, `*_roi.png`.

## Abhängigkeiten

Python 3 mit `numpy`, `scipy`, `opencv-python`, `scikit-image`.

```bash
pip install numpy scipy opencv-python scikit-image
```
