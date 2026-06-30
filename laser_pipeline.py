"""
laser_pipeline.py
=================
Vollständige Vorverarbeitung: Laserlinie im Referenzframe (Frame 0) erkennen,
als subpixelgenaue 1px-Linie fitten und eine ROI extrahieren.

Ablauf (Schritt für Schritt; ausführliche Begründung in doku_vorverarbeitung.tex,
Herleitung/verworfene Ansätze in ENTWICKLUNGSWEG_LASERLINIE.md):

  Schritt 1  Laden + Graustufen
  Schritt 2  Multi-Scale-Steger-Ridge-Detektion   -> alle Linienpixel + Stärke |λ_min|
  Schritt 3  Otsu-Kern + 2D-Dilation + P20-Filter  -> "Steger4": saubere, dichte Pixelmenge
  Schritt 4  Skelettierung + Pfadverfolgung        -> geordnete Teilstücke
                                                      (Trennung exakt an den großen Lücken)
  Schritt 5  Stärke-gewichteter Querschnitts-Schwerpunkt + Spline je Teilstück
                                                      (die starken/roten Pixel bestimmen die Lage)
  Schritt 6  Hermite-Brücke zwischen Teilstücken + lineare Endverlängerung
  Schritt 7  Ausgabe: 1px-Linie, Bogenlänge s, Normalen (Mess-Grundlage) als npz + Kontrollbild

Die senkrechte Verschiebung dieser Linie (Eis vs. Referenz) über der Bogenlänge s
ist die spätere Eisdicke; die Normalen geben die Messrichtung.

Rechenzeit: Multi-Scale-Steger ~40 s/Bild, der Fit < 1 s. Einmalige Kalibrierung je Setup.
"""

# ── Bibliotheken ──────────────────────────────────────────────────────────
from pathlib import Path             # plattformunabhängige Pfade (raw/, output/)
from collections import deque        # O(1)-Warteschlange für die Breitensuche (BFS)
import gc                            # manueller Garbage Collector (großer Speicher je Bild)
import time                          # Laufzeitmessung pro Bild
import cv2                           # OpenCV: Laden, Dilation, Otsu, Zeichnen
import numpy as np                   # gesamte numerische Array-Arithmetik
from scipy.ndimage import gaussian_filter          # Gauß-Ableitungen für die Hesse-Matrix
from scipy.interpolate import splprep, splev        # parametrischer Glättungs-Spline (Fit)
from scipy.spatial import cKDTree                    # schnelle Nächste-Nachbarn-Suche (Pixel->Skelett)
from skimage.morphology import skeletonize          # 1px-Skelett der Pixelmenge

# ── Pfade ─────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent       # Verzeichnis dieser Datei (= pre processing/Bild 0)
RAW    = HERE / "raw"                 # Eingabe: Roh-TIFs
OUTPUT = HERE / "output"             # Ausgabe: npz + Visualisierungen
OUTPUT.mkdir(parents=True, exist_ok=True)   # Ausgabeordner anlegen, falls nicht vorhanden

# ── Detektion (Steger) ────────────────────────────────────────────────────
SIGMA_B            = 15.0              # px Hintergrund-Gauß (DoG-Bandpass)
#SIGMA_SKALEN       = [1.0, 1.5, 3.0, 6.0]  # px Linien-Gauß je Skala (scharf … diffus)
SIGMA_SKALEN       = [1.0,  8.0]  # px Linien-Gauß je Skala (scharf … diffus)
LAMBDA_FAKTOR      = 0.02             # relative Mindest-Ridge-Stärke
PERCENTIL_FALLBACK = 60               # Otsu-Fallback, falls keine klare Trennung
DILATION_RADIUS    = 10               # px 2D-Dilation um den Otsu-Kern
P20_PERCENTIL      = 20               # schwächste % der NEU dazugekommenen Pixel verwerfen

# ── Fit ───────────────────────────────────────────────────────────────────
LUECKEN_SCHLIESSEN = 8     # px Dilationsradius: Band schließen (klein -> wenig Versatz)
MIN_KOMPONENTE     = 15    # px: kleinere Skelett-Fragmente verwerfen (Rauschen)
MIN_TEILSTUECK     = 60    # px: kurze Teilstücke verwerfen (instabile Tangente -> Knick)
MAX_BRUECKE        = 400   # px: max. Lücke, über die zwei Teilstücke verbunden werden
GEWICHT_EXP        = 4.0   # Stärke-Exponent für den Querschnitts-Schwerpunkt (rote Pixel dominieren)
MAX_VERSATZ        = 10    # px: max. Schwerpunkt-Versatz zur Skelettmitte (kappt ferne Spurs)
SMOOTH_PX          = 2.0   # Glättung s = SMOOTH_PX * N je Teilstück
VERLAENGERN_PX     = 200   # px lineare Verlängerung je Ende
TANGENTEN_FIT      = 25    # px für die gemittelte Rand-/Endsteigung

# ── Ausgabe ───────────────────────────────────────────────────────────────
NORMALE_STEP       = 120   # px-Abstand der gezeichneten Normalen (0 = aus)
NORMALE_LEN        = 30    # px Länge der gezeichneten Normalen


# ════════════════════════════════════════════════════════════════════════
#  Hilfsfunktion
# ════════════════════════════════════════════════════════════════════════
def kreiskernel(r):
    """Kreisförmiges Strukturelement (Radius r) für die Dilation."""
    d = 2 * r + 1                     # Kantenlänge des quadratischen Kernels (ungerade -> Mittelpixel)
    k = np.zeros((d, d), np.uint8)    # leeres Kernel-Bild
    cv2.circle(k, (r, r), r, 1, -1)   # gefüllten Kreis (Mittelpunkt (r,r), Radius r) auf 1 setzen
    return k                          # kreisförmige 0/1-Maske als Strukturelement


# ════════════════════════════════════════════════════════════════════════
#  Schritt 2 – Multi-Scale-Steger-Ridge-Detektion
# ════════════════════════════════════════════════════════════════════════
def hesse_dog(img_f, sigma_s, sigma_b):
    """Hesse-Ableitungen als Difference-of-Gaussians: G_sigma_s - G_sigma_b.
    Wirkt zugleich als Bandpass gegen breite Reflexe (Kanalwand, Endplatte)."""
    def d(order):
        # Eine partielle Ableitung als DoG: feine Skala (sigma_s) minus grobe (sigma_b).
        # scipy-order ist (Achse0=y, Achse1=x); die Differenz unterdrückt breite Strukturen.
        return (gaussian_filter(img_f, sigma_s, order=order) -
                gaussian_filter(img_f, sigma_b, order=order))
    # Rückgabe in der Reihenfolge Lxx, Lyy, Lxy, Lx, Ly:
    #   (0,2)=∂²/∂x²=Lxx, (2,0)=∂²/∂y²=Lyy, (1,1)=∂²/∂x∂y=Lxy, (0,1)=∂/∂x=Lx, (1,0)=∂/∂y=Ly
    return d((0,2)), d((2,0)), d((1,1)), d((0,1)), d((1,0))


def steger_multiscale(img_f):
    """Steger (1998): ein Pixel ist Linienrücken, wenn der kleinere Hesse-
    Eigenwert lambda_min stark negativ ist (starke Querkrümmung) UND der
    Subpixel-Offset |t| <= 0.5 entlang des Eigenvektors liegt.
    Über mehrere Skalen; pro Pixel gewinnt die Skala mit stärkstem |lambda_min|.
    -> Maske aller Ridge-Pixel + deren Stärke. float32 + del gegen Speicherlast."""
    H, W  = img_f.shape                          # Bildhöhe/-breite
    A_min = LAMBDA_FAKTOR * img_f.max()          # absolute Mindeststärke, skaliert mit Bildhelligkeit
    st    = np.zeros((H, W), dtype=np.float32)   # bisher stärkstes |λ_min| je Pixel (Skalen-Maximum)
    maske = np.zeros((H, W), dtype=bool)         # bisher als Ridge erkannte Pixel
    for sigma_s in SIGMA_SKALEN:                 # über alle Linienbreiten-Skalen iterieren
        schwelle = -A_min / (sigma_s ** 2)       # Steger-Schwelle: bei größerer Skala flacher (1/σ²)
        Lxx, Lyy, Lxy, Lx, Ly = hesse_dog(img_f, sigma_s, SIGMA_B)  # Hesse-Komponenten (DoG)
        spur    = Lxx + Lyy                       # Spur der Hesse-Matrix = λ1+λ2
        # Diskriminante der 2x2-Eigenwertformel; max(0,·) gegen Rundungs-Negativwerte
        diskr   = np.sqrt(np.maximum(0.0, (spur*.5)**2 - (Lxx*Lyy - Lxy**2)))
        lam_min = spur*.5 - diskr                 # kleinerer Eigenwert (stark negativ = Linienrücken)
        vx = Lxy.copy(); vy = lam_min - Lxx       # zugehöriger Eigenvektor (Querrichtung der Linie)
        norm = np.sqrt(vx**2 + vy**2) + 1e-12     # Betrag (Epsilon gegen Division durch 0)
        vx /= norm; vy /= norm                    # Eigenvektor auf Einheitslänge normieren
        t = -(Lx*vx + Ly*vy) / (lam_min - 1e-12)  # Subpixel-Offset = -(∇L·n)/λ_min entlang n
        m = (lam_min < schwelle) & (np.abs(t) <= 0.5)  # Ridge-Bedingung: stark gekrümmt UND Mitte im Pixel
        s = np.abs(lam_min).astype(np.float32)    # Ridge-Stärke = |λ_min|
        besser = m & (s > st)                     # Pixel, die diese Skala stärker erkennt als bisher
        st[besser] = s[besser]; maske[besser] = True   # Skalen-Maximum + Maske aktualisieren
        # Zwischenarrays sofort freigeben (5312x4592 float -> ~100 MB je Array)
        del Lxx, Lyy, Lxy, Lx, Ly, spur, diskr, lam_min, vx, vy, norm, t, m, s, besser
    return maske, st                              # Maske aller Ridge-Pixel + Stärkekarte


# ════════════════════════════════════════════════════════════════════════
#  Schritt 3 – Steger4-Filter: Otsu-Kern + 2D-Dilation + P20
# ════════════════════════════════════════════════════════════════════════
def otsu_filter(maske, staerke):
    """Otsu-Schwelle auf |lambda_min| -> nur die stärksten Flanken-Pixel
    ("Steger3"-Kern). Perzentil-Fallback, falls Otsu nicht klar trennt."""
    if not maske.any():                          # keine Ridge-Pixel -> nichts zu filtern
        return maske
    werte = staerke[maske]                       # Stärkewerte nur der Ridge-Pixel
    w_min, w_max = werte.min(), werte.max()      # Wertebereich für die Normierung
    if w_max <= w_min:                           # alle gleich -> Otsu nicht definiert
        return maske
    norm = ((werte - w_min) / (w_max - w_min) * 255).astype(np.uint8)  # auf 0..255 skalieren
    otsu_val, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu-Schwelle
    schwelle = w_min + (otsu_val / 255.0) * (w_max - w_min)   # Schwelle in die Originalskala zurück
    anteil   = (werte >= schwelle).mean()        # Anteil der behaltenen Pixel
    if 0.05 < anteil < 0.95:                      # nur wenn Otsu sinnvoll trennt (nicht fast alles/nichts)
        return maske & (staerke >= schwelle)     # Kern = Pixel oberhalb der Otsu-Schwelle
    fallback = np.percentile(werte, PERCENTIL_FALLBACK)  # sonst festes Perzentil als Schwelle
    return maske & (staerke >= fallback)


def steger4(img_f):
    """Detektion gesamt -> finale Pixelmenge + Stärke.
    1. Multi-Scale-Steger: alle Ridge-Pixel.
    2. Otsu-Kern: saubere, sichere Linienpixel (immer behalten).
    3. 2D-Dilation ±DILATION_RADIUS um den Kern: holt gesättigte Lasermitte und
       schwache Endpunkte zurück, NUR in der Nähe sicherer Pixel (kein fernes Rauschen).
    4. Von den NEU dazugekommenen Pixeln die schwächsten P20_PERCENTIL % verwerfen
       (die "extremgelben" Ausreißer am Rand der Dilation)."""
    maske_alle, staerke = steger_multiscale(img_f)            # 1. alle Ridge-Pixel + Stärke
    kern = otsu_filter(maske_alle, staerke)                   # 2. sicherer Otsu-Kern
    dilated = cv2.dilate(kern.astype(np.uint8), kreiskernel(DILATION_RADIUS)) > 0  # 3. Kern aufweiten
    erweitert = maske_alle & dilated                          #    Ridge-Pixel in Kernnähe zulassen
    neu = erweitert & ~kern                                   #    nur die NEU dazugekommenen Pixel
    if neu.any():
        schwelle = np.percentile(staerke[neu], P20_PERCENTIL) # 4. P20-Schwelle nur auf den neuen Pixeln
        neu = neu & (staerke >= schwelle)                     #    schwächste 20% der neuen verwerfen
    return (kern | neu), staerke                              # finale Pixelmenge (Kern + gefilterte neue)


# ════════════════════════════════════════════════════════════════════════
#  Schritt 4 – Skelettierung + Pfadverfolgung -> geordnete Teilstücke
# ════════════════════════════════════════════════════════════════════════
def verkette(paths, max_bruecke):
    """Teilstücke richtungsrichtig ordnen (nicht zusammenfügen): längstes zuerst,
    dann iterativ das nächstgelegene Stück vorn/hinten anhängen (mit Flip).
    Lücken > max_bruecke werden NICHT verbunden (keine fremden Fragmente)."""
    paths = sorted(paths, key=len, reverse=True)  # längstes Teilstück zuerst (stabiler Anker)
    kette = [paths[0]]                            # Ergebnis-Reihenfolge, beginnt mit dem längsten
    rest  = paths[1:]                             # noch einzuordnende Teilstücke

    def d(a, b):
        return float(np.hypot(a[0]-b[0], a[1]-b[1]))  # euklidischer Abstand zweier Endpunkte

    while rest:                                   # bis alle einsortiert (oder zu weit entfernt)
        front, back = kette[0][0], kette[-1][-1]  # aktuelle freie Enden der Kette (Anfang/Ende)
        best = None                               # beste (kürzeste) Andock-Option
        for k, P in enumerate(rest):              # jedes Rest-Teilstück testen
            # 4 Andock-Varianten: an Ende/Anfang der Kette, jeweils ohne/mit Flip des Teilstücks
            for c in [(d(back, P[0]),  k, "back",  False),
                      (d(back, P[-1]), k, "back",  True),
                      (d(front, P[-1]), k, "front", False),
                      (d(front, P[0]),  k, "front", True)]:
                if best is None or c[0] < best[0]:  # kleinsten Endpunktabstand merken
                    best = c
        if best is None or best[0] > max_bruecke:  # nächstes Stück zu weit weg -> abbrechen
            break
        _, k, wohin, flip = best                  # beste Option auflösen
        P = rest.pop(k)                           # Teilstück aus der Restliste nehmen
        if flip:
            P = P[::-1]                            # Laufrichtung umkehren, damit die Enden passen
        kette.append(P) if wohin == "back" else kette.insert(0, P)  # hinten anhängen oder vorn einfügen
    return kette                                  # geordnete Liste der Teilstücke


def skelett_teilstuecke(xs, ys, schliessen):
    """Pixelmenge -> geordnete Liste der 1px-Teilstücke.
    Kleine Lücken (< 2*schliessen) werden zum Band geschlossen, die GROSSE Lücke
    bleibt offen -> getrennte Komponenten, deren Enden genau am Lückenrand liegen
    (in der großen Lücke also keine Punkte). Pro Komponente der längste
    geodätische Pfad (Doppel-BFS) = geordnetes Teilstück; kurze Stücke verworfen."""
    x0, x1 = xs.min(), xs.max()                   # Bounding-Box der Pixelmenge (x)
    y0, y1 = ys.min(), ys.max()                   # Bounding-Box (y)
    pad = schliessen + 2                          # Rand, damit die Dilation nicht abgeschnitten wird
    m = np.zeros(((y1-y0)+2*pad+1, (x1-x0)+2*pad+1), np.uint8)  # leeres Maskenbild (auf Bbox zugeschnitten)
    m[ys - y0 + pad, xs - x0 + pad] = 1           # Pixel in lokale Koordinaten setzen
    m = cv2.dilate(m, kreiskernel(schliessen))    # kleine Lücken schließen (große Lücke bleibt offen)
    skel = skeletonize(m > 0)                      # auf 1px-Mittellinie reduzieren

    sy, sx = np.where(skel)                        # Koordinaten aller Skelettpixel
    if len(sy) < 4:                                # zu wenige Punkte -> keine Linie
        return None
    coords = np.stack([sy, sx], 1)                 # (N,2)-Array (y,x) der Skelettpixel
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(coords)}  # Koordinate -> Knotenindex
    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]      # 8er-Nachbarschaft
    adj = [[] for _ in range(len(coords))]         # Adjazenzliste (Graph über die Skelettpixel)
    for i, (a, b) in enumerate(coords):            # für jeden Skelettknoten ...
        for da, db in nb:                          # ... alle 8 Nachbarpositionen prüfen
            j = index.get((int(a)+da, int(b)+db))  # existiert dort ein Skelettpixel?
            if j is not None:
                adj[i].append(j)                   # Kante eintragen

    def bfs(start, erlaubt=None):
        # Breitensuche ab 'start'; liefert den entferntesten Knoten + Distanz-/Eltern-Karten.
        # 'erlaubt' beschränkt die Suche auf eine Komponente (Knotenmenge).
        dist = {start: 0}; parent = {start: -1}
        q = deque([start]); last = start
        while q:
            u = q.popleft(); last = u              # 'last' ist am Ende der am weitesten entfernte Knoten
            for v in adj[u]:
                if v not in dist and (erlaubt is None or v in erlaubt):
                    dist[v] = dist[u] + 1; parent[v] = u; q.append(v)
        return last, dist, parent

    besucht = set(); paths = []                    # bereits abgearbeitete Knoten / Ergebnis-Pfade
    for s in range(len(coords)):                   # alle Knoten durchgehen (mehrere Komponenten möglich)
        if s in besucht:
            continue
        _, dist0, _ = bfs(s)                       # 1. BFS: erfasst die ganze Komponente von s aus
        komp = set(dist0.keys()); besucht |= komp  # Komponente markieren
        if len(komp) < MIN_KOMPONENTE:             # zu kleine Fragmente (Rauschen) verwerfen
            continue
        a, _, _ = bfs(next(iter(komp)), komp)      # 2. BFS: entferntester Knoten a (ein Linienende)
        b, _, parent = bfs(a, komp)                # 3. BFS ab a: b = anderes Ende, parent rekonstruiert Pfad
        pf = []; u = b
        while u != -1:                             # Pfad b -> a über die Eltern zurückverfolgen
            pf.append(u); u = parent[u]
        yx = coords[pf[::-1]]                       # Reihenfolge a -> b (geordnetes Teilstück), (y,x)
        # zurück in globale Bildkoordinaten (x,y) und lokalen Pad/Bbox-Offset abziehen
        xy = np.stack([yx[:, 1] + x0 - pad, yx[:, 0] + y0 - pad], 1).astype(float)
        # nur Teilstücke ab Mindestlänge behalten (kurze -> instabile Tangente -> Knick)
        if float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))) >= MIN_TEILSTUECK:
            paths.append(xy)

    return verkette(paths, MAX_BRUECKE) if paths else None  # Teilstücke ordnen (oder None)


# ════════════════════════════════════════════════════════════════════════
#  Schritt 5 – stärke-gewichteter Querschnitts-Schwerpunkt + Spline
# ════════════════════════════════════════════════════════════════════════
def laser_punkte(segs, pix_xy, pix_w):
    """Pro Skelettpunkt der mit |lambda_min|^GEWICHT_EXP gewichtete Schwerpunkt der
    zugeordneten Steger4-Pixel -> die starken (roten) Pixel, die exakt auf dem
    Laser liegen, bestimmen die Linienlage (nicht die geometrische Bandmitte, die
    von den schwachen Seiten-Spurs verzogen würde). Versatz auf MAX_VERSATZ gekappt."""
    offs   = np.concatenate([[0], np.cumsum([len(s) for s in segs])])  # Index-Grenzen je Teilstück
    allpts = np.vstack(segs)                       # alle Skelettpunkte in einem Array (für KDTree)
    nn = cKDTree(allpts).query(pix_xy)[1]          # jedem Steger4-Pixel den nächsten Skelettpunkt zuordnen
    we = pix_w ** GEWICHT_EXP                       # Gewicht = Stärke^p (starke Pixel dominieren stark)
    refined, gewichte = [], []                      # Ergebnis: korrigierte Punkte + Gewichte je Teilstück
    for si, s in enumerate(segs):                   # jedes Teilstück einzeln
        lo, hi = offs[si], offs[si+1]               # globaler Indexbereich dieses Teilstücks
        sel = (nn >= lo) & (nn < hi)                # Pixel, die diesem Teilstück zugeordnet sind
        loc = nn[sel] - lo                          # lokale Skelettpunkt-Indizes (0..len(s)-1)
        pm, wm = pix_xy[sel], we[sel]               # Positionen + Gewichte dieser Pixel
        sx = np.zeros(len(s)); sy = np.zeros(len(s)); sw = np.zeros(len(s))  # Akkumulatoren je Skelettpunkt
        np.add.at(sx, loc, pm[:, 0] * wm)           # Σ(gewichtetes x) je Skelettpunkt
        np.add.at(sy, loc, pm[:, 1] * wm)           # Σ(gewichtetes y)
        np.add.at(sw, loc, wm)                       # Σ(Gewicht)
        R = s.astype(float).copy()                  # Startwert = geometrische Skelettlage
        gut = sw > 0                                 # Skelettpunkte mit zugeordneten Pixeln
        R[gut, 0] = sx[gut] / sw[gut]               # gewichteter x-Schwerpunkt
        R[gut, 1] = sy[gut] / sw[gut]               # gewichteter y-Schwerpunkt
        shift = R - s                                # Versatz Schwerpunkt - Skelettmitte
        mag = np.hypot(shift[:, 0], shift[:, 1])     # Betrag des Versatzes je Punkt
        zu = mag > MAX_VERSATZ                        # zu große Versätze (ferne Seiten-Spurs)
        shift[zu] *= (MAX_VERSATZ / mag[zu])[:, None] # auf MAX_VERSATZ kürzen (Richtung bleibt)
        R = s + shift                                 # gekappter, korrigierter Punkt
        # Fit-Gewicht je Punkt: belegte Punkte = Σ Gewicht; leere bekommen ein winziges Gewicht
        Wt = np.where(gut, sw, (sw[gut].max() if gut.any() else 1.0) * 1e-3)
        refined.append(R); gewichte.append(Wt)
    return refined, gewichte                          # korrigierte Punkte + Fit-Gewichte je Teilstück


def fit_teilstueck(R, Wt):
    """Ein Teilstück eigenständig als gewichteter Glättungs-Spline -> dichte (n,2)."""
    X, Y = R[:, 0], R[:, 1]                          # x/y-Koordinaten des Teilstücks
    N = len(X)                                        # Anzahl Stützpunkte
    k = min(3, N - 1)                                 # Spline-Grad (kubisch, bei wenig Punkten kleiner)
    w = np.sqrt(Wt / (Wt.max() + 1e-12))              # Spline-Gewichte (splprep erwartet ~1/σ, hier √Stärke)
    tck, _ = splprep([X, Y], w=w, s=SMOOTH_PX * N, k=k)  # parametrischer Glättungs-Spline (x(t),y(t))
    laenge = float(np.sum(np.hypot(np.diff(X), np.diff(Y))))  # grobe Bogenlänge des Teilstücks
    n = int(np.clip(laenge * 1.5, 50, 8000))          # Abtastdichte ~1.5 Punkte/px (begrenzt)
    xx, yy = splev(np.linspace(0, 1, n), tck)         # Spline gleichmäßig in t abtasten
    return np.stack([np.asarray(xx), np.asarray(yy)], 1)  # dichte (n,2)-Linie


def fitte_segmente(segs, pix_xy, pix_w):
    """Schritt 5 gesamt: jedes Teilstück eigenständig gewichtet fitten.
    -> Liste dichter (n,2)-Arrays, je ein Teilstück, unverbunden."""
    refined, gewichte = laser_punkte(segs, pix_xy, pix_w)        # gewichtete Schwerpunkte je Teilstück
    return [fit_teilstueck(R, Wt) for R, Wt in zip(refined, gewichte)]  # je Teilstück ein Spline


# ════════════════════════════════════════════════════════════════════════
#  Schritt 6 – Hermite-Brücke + lineare Endverlängerung
# ════════════════════════════════════════════════════════════════════════
def tangente(dense, am_ende, fit_px):
    """Einheits-Tangente in Laufrichtung am Anfang/Ende eines dichten Stücks."""
    k = min(fit_px, len(dense) - 1)                   # Stützweite (über fit_px Punkte mitteln)
    v = (dense[-1] - dense[-1 - k]) if am_ende else (dense[k] - dense[0])  # Sekante am Ende/Anfang
    return v / (np.hypot(*v) + 1e-12)                 # auf Einheitslänge normieren


def hermite_bruecke(p0, t0, p1, t1):
    """Kubischer Hermite-Übergang p0->p1 mit Laufrichtungs-Tangenten t0,t1:
    übernimmt Position UND Tangente an beiden Lückenrändern exakt -> knickfrei,
    ohne die Teilstück-Fits zu verändern. Endpunkte ausgeschlossen."""
    d = float(np.hypot(*(p1 - p0)))                   # Lückenweite (skaliert die Tangentenlänge)
    n = max(2, int(d))                                 # ~1 Brückenpunkt je Pixel
    m0, m1 = t0 * d, t1 * d                            # Hermite-Tangenten (mit Lückenweite skaliert)
    s = np.linspace(0, 1, n)[1:-1]                     # Parameter 0..1 ohne die Endpunkte (gehören den Teilstücken)
    h00 = 2*s**3 - 3*s**2 + 1                          # Hermite-Basis: Position p0
    h10 = s**3 - 2*s**2 + s                            # Hermite-Basis: Tangente m0
    h01 = -2*s**3 + 3*s**2                             # Hermite-Basis: Position p1
    h11 = s**3 - s**2                                  # Hermite-Basis: Tangente m1
    x = h00*p0[0] + h10*m0[0] + h01*p1[0] + h11*m1[0]  # x-Kurve der Brücke
    y = h00*p0[1] + h10*m0[1] + h01*p1[1] + h11*m1[1]  # y-Kurve der Brücke
    return np.stack([x, y], 1)                          # (n-2, 2) Brückenpunkte


def verbinde(denses):
    """Enden per Hermite verbinden, OHNE die Teilstück-Fits anzufassen.
    -> PTS (M,2), ist_bruecke (M,) bool (True = interpolierte Lücke)."""
    teile, marken = [denses[0]], [np.zeros(len(denses[0]), bool)]  # erstes Teilstück (keine Brücke)
    for i in range(1, len(denses)):                    # jede folgende Lücke überbrücken
        A, B = denses[i-1], denses[i]                  # vorheriges und nächstes Teilstück
        br = hermite_bruecke(A[-1], tangente(A, True, TANGENTEN_FIT),   # Brücke: Ende von A ...
                             B[0],  tangente(B, False, TANGENTEN_FIT))  # ... zu Anfang von B (tangentenstetig)
        teile.append(br);  marken.append(np.ones(len(br), bool))   # Brückenpunkte (als Brücke markiert)
        teile.append(B);   marken.append(np.zeros(len(B), bool))   # echtes Teilstück B (keine Brücke)
    return np.vstack(teile), np.concatenate(marken)    # zusammenhängende Linie + Brücken-Flag


def endverlaengerung(PTS, laenge, fit_px):
    """Linien-Enden mit der jeweiligen Endpunkt-Steigung um 'laenge' px verlängern."""
    t0 = tangente(PTS, False, fit_px)                  # Tangente am Linienanfang
    t1 = tangente(PTS, True,  fit_px)                  # Tangente am Linienende
    t = np.arange(1, laenge + 1)[:, None]              # Schrittweiten 1..laenge (Spaltenvektor)
    vorn = PTS[0]  - t0 * t                            # geradlinige Verlängerung vor dem Anfang
    hint = PTS[-1] + t1 * t                            # geradlinige Verlängerung nach dem Ende
    return np.vstack([vorn[::-1], PTS, hint]), len(vorn)  # verlängerte Linie + Anzahl Verlängerungspunkte


# ════════════════════════════════════════════════════════════════════════
#  Schritt 7 – Bogenlänge, Normalen, Rasterung, Ausgabe
# ════════════════════════════════════════════════════════════════════════
def bogenlaenge_und_normalen(voll):
    """Kumulierte Bogenlänge s und Einheits-Normalen (Messrichtung) je Linienpunkt."""
    tang = np.gradient(voll, axis=0)                   # Tangentenvektor je Punkt (zentrale Differenz)
    nl = np.hypot(tang[:, 0], tang[:, 1]) + 1e-12      # Tangentenbetrag (Epsilon gegen 0)
    normalen = np.stack([-tang[:, 1] / nl, tang[:, 0] / nl], 1)  # Normale = Tangente um 90° gedreht, normiert
    # Bogenlänge = kumulierte Abstände aufeinanderfolgender Punkte (Start = 0)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(voll[:, 0]),
                                                np.diff(voll[:, 1])))])
    return s, normalen                                 # s (Messachse) und Normalen (Messrichtung)


def raster_1px(P, H, W):
    """Dichte Linie -> eindeutige, geordnete 1px-Pixel (gerundet, geklippt)."""
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)  # x runden + ins Bild klippen
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)  # y runden + ins Bild klippen
    key = yi.astype(np.int64) * W + xi                 # eindeutiger Schlüssel je Pixel (y*W+x)
    _, ui = np.unique(key, return_index=True)          # Duplikate entfernen (erstes Vorkommen behalten)
    ui = np.sort(ui)                                    # ursprüngliche Reihenfolge wiederherstellen
    return xi[ui], yi[ui]                               # geordnete, eindeutige Pixelkoordinaten


def main():
    tifs = sorted(RAW.glob("*.tif"))                   # alle Roh-TIFs einsammeln
    print(f"Gefundene Bilder: {len(tifs)}\n")

    for tif in tifs:                                    # jedes Bild einzeln verarbeiten
        t0 = time.perf_counter()                       # Startzeit (Laufzeitmessung)
        print(tif.name)

        # Schritt 1 – Laden + Graustufen
        img = cv2.imread(str(tif), cv2.IMREAD_GRAYSCALE)  # direkt als Graustufe laden (Monochromsensor)
        if img is None:
            print("  FEHLER: nicht ladbar."); continue  # defektes/fehlendes Bild überspringen
        H, W = img.shape                               # Bildmaße
        img_f = img.astype(np.float32)                 # float für die Ableitungs-Arithmetik

        # Schritt 2+3 – Detektion -> Steger4-Pixel
        maske, staerke = steger4(img_f)                # finale Pixelmenge + Stärkekarte
        ys, xs = np.where(maske)                        # Pixelkoordinaten
        w = staerke[ys, xs].astype(float)              # zugehörige Ridge-Stärken (Gewicht für den Fit)
        print(f"  Steger4: {len(xs)} px  [{time.perf_counter()-t0:.1f}s]")

        # Schritt 4 – Teilstücke
        segs = skelett_teilstuecke(xs, ys, LUECKEN_SCHLIESSEN)  # geordnete 1px-Teilstücke
        if not segs:
            print("  kein Teilstück."); continue        # keine Linie erkannt -> nächstes Bild
        pix_xy = np.stack([xs, ys], 1).astype(float)   # Steger4-Pixel als (x,y)-Array

        # Schritt 5 – stärke-gewichteter Fit je Teilstück
        denses = fitte_segmente(segs, pix_xy, w)       # je Teilstück ein gewichteter Spline

        # Schritt 6 – verbinden + verlängern
        PTS, ist_br = verbinde(denses)                 # Teilstücke per Hermite-Brücke verbinden
        voll, n_ext = endverlaengerung(PTS, VERLAENGERN_PX, TANGENTEN_FIT)  # Enden geradlinig verlängern
        ist_br_voll = np.concatenate([np.zeros(n_ext, bool), ist_br,        # Brücken-Flag auf die ...
                                      np.zeros(n_ext, bool)])                # ... verlängerte Linie ausdehnen

        # Schritt 7 – Bogenlänge, Normalen, Ausgabe
        s, normalen = bogenlaenge_und_normalen(voll)   # Messachse s + Messrichtungen
        L = float(s[-1])                               # Gesamt-Bogenlänge

        # Ergebnis als Datensatz (Grundlage der Eisdickenmessung)
        np.savez_compressed(
            OUTPUT / f"{tif.stem}_laserlinie.npz",
            x=voll[:, 0].astype(np.float32), y=voll[:, 1].astype(np.float32),  # Linienkoordinaten
            s=s.astype(np.float32), nx=normalen[:, 0].astype(np.float32),       # Bogenlänge + Normale x
            ny=normalen[:, 1].astype(np.float32),                              # Normale y
            ist_bruecke=ist_br_voll, bogenlaenge=np.float32(L), H=H, W=W)       # Flags + Maße

        # Visualisierung
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    # Graubild -> BGR zum Einfärben
        rgb[ys, xs] = (90, 90, 0)                            # Steger4-Pixel dezent
        kern = voll[n_ext:len(voll)-n_ext]             # gefittete Linie ohne die Verlängerungen
        xk, yk = raster_1px(kern[~ist_br], H, W); rgb[yk, xk] = (0, 0, 255)   # rot
        if ist_br.any():
            xb, yb = raster_1px(kern[ist_br], H, W); rgb[yb, xb] = (255, 0, 255)  # magenta
        xa, ya = raster_1px(voll[:n_ext], H, W); rgb[ya, xa] = (0, 220, 255)  # Verlängerung vorn (gelb)
        xe, ye = raster_1px(voll[-n_ext:], H, W); rgb[ye, xe] = (0, 220, 255)  # Verlängerung hinten
        if NORMALE_STEP > 0:                            # Normalen in festem Bogenlängen-Abstand zeichnen
            naechste = 0.0
            for i in range(len(voll)):
                if s[i] >= naechste:                    # nächste Markierung erreicht?
                    nx, ny = normalen[i]
                    p1 = (int(voll[i,0]-nx*NORMALE_LEN), int(voll[i,1]-ny*NORMALE_LEN))  # Normale -Seite
                    p2 = (int(voll[i,0]+nx*NORMALE_LEN), int(voll[i,1]+ny*NORMALE_LEN))  # Normale +Seite
                    cv2.line(rgb, p1, p2, (255, 120, 0), 1)
                    naechste += NORMALE_STEP            # nächste Markierung weiterschalten
        cv2.imwrite(str(OUTPUT / f"{tif.stem}_fit.png"), rgb)

        # Hinweis: Die eigentliche Mess-ROI (gerundetes adaptives Band) liefert
        # crop_roi.py aus dem _laserlinie.npz; hier wird bewusst kein ROI-Bild
        # mehr gespeichert.
        print(f"  {len(segs)} Teilstück(e)  Bogenlänge {L:.0f}px  "
              f"Brücke {int(ist_br.sum())}px  "
              f"-> _fit.png + _laserlinie.npz  "
              f"[{time.perf_counter()-t0:.1f}s]")

        del img, img_f, maske, staerke, rgb            # große Arrays freigeben
        gc.collect()                                    # Speicher vor dem nächsten Bild zurückgeben


if __name__ == "__main__":
    main()
