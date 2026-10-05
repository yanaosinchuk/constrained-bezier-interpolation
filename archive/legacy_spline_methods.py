import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Funktion zum Berechnen der ai, bi, ci, di. Rückgabe der Koeffizienten für jedes Intervall und alle x-Werte
# Benötigt alle Stützpunkte und Periodenlänge
def kubiperispline(x, y, peri):
    x_all = np.append(x, peri)
    y_all = np.append(y, y[0])
    # hs werden berechnet
    h = np.diff(x_all)
    anzx = len(x)

    # Koeffizientenmatrix wird angelegt und im Anschluss befüllt
    koeffmatrix = np.zeros((anzx, anzx))
    for i in range(anzx-2):
        # obere Nebendiagonalen mit h[i+1] füllen
        koeffmatrix[i+1][i+2] = h[i+1]
        # untere Nebendiagonalen mit h[i] füllen
        koeffmatrix[i+1][i] = h[i]
        # Hauptdiagonale mit 2*(h[i]+h[i+1]) füllen
        koeffmatrix[i+1][i+1] = 2*(h[i]+h[i+1])

    # füllen der ersten Zeile
    # erste Spalte
    koeffmatrix[0][0] = 2 * (h[-1] + h[0])
    # zweite Spalte
    koeffmatrix[0][1] = h[0]
    # letzte Spalte
    koeffmatrix[0][-1] = h[-1]

    # füllen der letzten Zeile
    # erste Spalte
    koeffmatrix[-1][0] = h[-1]
    # vorletzte Spalte
    koeffmatrix[-1][-2] = h[-2]
    # letzte Spalte
    koeffmatrix[-1][-1] = 2 * (h[-2] + h[-1])
    # folgender print-Befehl kann angeschaltet werden, um die Koeffizientenmatrix auszugeben
    #print(koeffmatrix)

    # Erstellen der rechten Seite
    rechteseite = np.zeros(anzx)
    # Belegen der rechten Seite mit Werten für die mittleren Zeilen
    for i in range(1, anzx-1):
        rechteseite[i] = (6/h[i]) * (y_all[i+1] - y_all[i]) - (6/h[i-1]) * (y[i] - y[i-1])

    # Belegen der ersten Zeile der rechten Seite
    rechteseite[0] = (6/h[0]) * (y[1] - y[0]) - (6/h[-1]) * (y[0] - y[-1])
    # Belegen der letzten Zeile der rechten Seite
    rechteseite[-1] = (6/h[-1]) * (y[0] - y[-1]) - (6/h[-2]) * ( y[-1] - y[-2])

    # Berechnung des LGS mit eigenem solver
    y2s = thomas_spezial_alg(koeffmatrix, rechteseite)
    #print(y2s)
    #y2s = np.linalg.solve(koeffmatrix, rechteseite)
    #print(y2s)

    # Anlegen der Koeffizientenarrays
    a = np.zeros(anzx)
    b = np.zeros(anzx)
    c = np.zeros(anzx)
    d = np.zeros(anzx)

    # Berechnen der Koeffizienten
    for i in range(anzx):
        b[i] = (1/2) * y2s[i]
        d[i] = y_all[i]

    for i in range(anzx-1):
        a[i] = (1/(6 * h[i])) * (y2s[i+1] - y2s[i])
        c[i] = (1/h[i]) * (y_all[i+1] - y_all[i]) - (1/6) * h[i] * (y2s[i+1] + 2 * y2s[i])

    a[-1] = (1 / (6 * h[-1])) * (y2s[0] - y2s[-1])
    c[-1] = (1 / h[-1]) * (y_all[-1] - y_all[-2]) - (1 / 6) * h[-1] * (y2s[0] + 2 * y2s[-1])

    return a, b, c, d, x_all

# Funktion für kubisches Polynom
def cubicpoly(x, xi, a, b, c, d):
    return a * (x -xi)**3 + b * (x - xi)**2 + c * (x - xi) + d

# Funktion zur Berechnung der x-Werte aus gegebenem Datum aus Excel
# Benötigt Liste mit Datum-Strings und Anfangsdatum als String
def zeitseitbeginn(datumstr, anfangsdatum):
    # Input wird in datetimeobjekte umgewandelt
    anfangs_datum_obj = datetime.strptime(anfangsdatum, "%Y-%m-%d %H:%M:%S")
    datum_obj = datetime.strptime(datumstr, "%Y-%m-%d %H:%M:%S")
    # Berechnung der Differenz eines Datums zum Anfangsdatum
    differenz = datum_obj - anfangs_datum_obj
    # Umrechnen der Differenz datetimeobjekte in Anzahl der Tage
    tage = differenz.total_seconds() / 60 / 60 / 24
    return tage

def thomas_alg(matrix, rs):
    n = len(matrix)
    # Anlegen der unteren, mittleren und oberen Diagonalen (gleich lang, weil 0 bei oberer und unterer für t.algo
    # hinzugefügt wird
    untere = np.zeros(n)
    diag = np.zeros(n)
    obere = np.zeros(n)

    for i in range(n):
        # Hauptdiagonale wird besetzt
        diag[i] = matrix[i][i]
        # Nebendiagonalen werden beide besetzt (Ausnutzen der Symmetrie)
        if i > 0:
            untere[i] = matrix[i][i - 1]
            obere[i - 1] = matrix[i][i - 1]

    # Vorwärtsdurchlauf (ci,di anlegen und ersten Wert der cis,dis füllen)
    ci = np.zeros(n)
    ci[0] = obere[0] / diag[0]
    di = np.zeros(n)
    di[0] = rs[0] / diag[0]

    # ci und di gemäß Thomas-Algorithmusvorschrift füllen
    for i in range(1, n):
        ci[i] = obere[i] / (diag[i] - ci[i - 1] * untere[i])

    for i in range(1, n):
        di[i] = (rs[i] - di[i - 1] * untere[i]) / (diag[i] - ci[i - 1] * untere[i])

    # Rückwärtsdurchlauf (xi anlegen und den letzten Wert der xi füllen)
    xi = np.zeros(n)
    xi[-1] = di[-1]

    # xi berechnen (rückwärtslaufende schleife)
    for i in range(n - 2, -1, -1):
        xi[i] = di[i] - ci[i] * xi[i + 1]

    return xi

# Funktion für das Lösen von LGS (Unterteilung in einen Block der mit normalem Thomas-Algorithmus gelöst werden kann
# und effiziente Berechnung des Rests)
def thomas_spezial_alg(matrix, rs):
    # Matrix ohne letzte Zeile und Spalte
    a = matrix[:-1,:-1]

    # Letzte Spalte ohne letzte Zeile
    d = matrix[:-1, -1]

    # Transponierter Vektor d (Symmetrie ausnutzen)
    dt = d.T
    # letze Zeile letzte Spalte letztes Element
    alpha = matrix[-1,-1]

    # rechte Seite ohne letztes Element
    b = rs[:-1]

    # letztes Element der rechten Seite
    beta =rs[-1]


    # a ist jetzt tridiagonale Matrix (Rest wird extra berechnet)
    v = thomas_alg(a,b)
    #v = np.linalg.solve(a,b)

    # statt langem Skalarprodukt mit vielen 0 Einträgen
    oben = dt[0]*v[0] + dt[-1]*v[-1]

    w = thomas_alg(a, d)
    #w = np.linalg.solve(a,d)
    # Äquivalent zu oben
    unten = dt[0] * w[0] + dt[-1] * w[-1]

    # gesuchten Vektor in zwei Teilen berechnen
    xzi =  (beta - oben)/(alpha - unten)
    xdach = v - xzi*w

    # zu einem Vektor wieder verbinden
    z = np.append(xdach, xzi)
    return z

# Funktion, um das richtige Intervall der Koeffizienten zu suchen (binäre Suche)
# Benötigt eine sortierte Liste (unsere Liste von Natur aus sortiert) und einen Wert, dessen Intervall wir suchen
def intervall_suche(x, x0):
    untere_grenze = 0
    obere_grenze = len(x) - 1

    # Solange obere und untere Grenze nicht vom selben Intervall sind, suchen wir weiter
    while obere_grenze - untere_grenze > 1:
        # Intervall wird halbiert
        k_mitte = (untere_grenze + obere_grenze) // 2  # immer abgerundet
        # Abfrage welche Grenze verschoben wird, um das Intervall einzugrenzen
        if x[untere_grenze] <= x0 < x[k_mitte]:
            obere_grenze = k_mitte
        else:
            untere_grenze = k_mitte

    # Falls ein Wert außerhalb der Grenzen abgefragt wird
    if not (x[untere_grenze] <= x0 <= x[untere_grenze + 1]):
        return "error"

    return untere_grenze

# Funktion, um einen Wert eines bestimmten Datums abzufragen
def y_suche(zeitstr, anfangstr, a, b, c, d, x_all):
    # 5 Zeilen analog zu zeitseitbeginn()
    anfangs_datum_obj = datetime.strptime(anfangstr, "%Y-%m-%d %H:%M:%S")
    datum_obj = datetime.strptime(zeitstr, "%Y-%m-%d %H:%M:%S")
    zeit_nach_peri = datum_obj.replace(year=anfangs_datum_obj.year)
    differenz = zeit_nach_peri - anfangs_datum_obj
    xwert = differenz.total_seconds() / 60 / 60 / 24

    # Abfrage, ob Wert kleiner 0 (Schaltjahrhandling zwingt diese Abfrage auf)
    # wenn ein Datum vor dem ersten Messwert erfragt wird.
    # z. B. erste Messung ist am 5.1.2010, frage nach Daten zwischen 31.12 und 5.1 egal welchen Jahres müssen
    # extra betrachtet werden
    if xwert < 0:
        zeit_nach_peri = datum_obj.replace(year=(anfangs_datum_obj.year)+1)
        differenz = zeit_nach_peri - anfangs_datum_obj
        xwert = differenz.total_seconds() / 60 / 60 / 24

    #print("test", xwert)
    # nach Umrechnen von Datum in Tages-Werte kann Intervallsuche beginnen
    gesuchtes_intervall = intervall_suche(x_all, xwert)

    # Wenn Intervall gefunden, kann mit dessen Koeffizienten der Wert berechnet werden
    gesuchtes_y = cubicpoly(xwert, x_all[gesuchtes_intervall], a[gesuchtes_intervall],
                                       b[gesuchtes_intervall], c[gesuchtes_intervall], d[gesuchtes_intervall])
    return xwert, gesuchtes_y

# alternative Möglichkeit für klassischen kubischen Spline
def normal_spline(x,y,peri):
    x_all = np.append(x, peri)
    h= np.diff(x_all)
    n = len(x)
    gr = n*4
    matrix = np.zeros((gr, gr))

    for i in range(n - 1):
        matrix[0 + 4 * i][0 + 4 * i] = h[i] ** 3
        matrix[0 + 4 * i][1 + 4 * i] = h[i] ** 2
        matrix[0 + 4 * i][2 + 4 * i] = h[i]
        matrix[0 + 4 * i][3 + 4 * i] = 1
        matrix[0 + 4 * i][7 + 4 * i] = -1

        matrix[1 + 4 * i][0 + 4 * i] = 3 * h[i] ** 2
        matrix[1 + 4 * i][1 + 4 * i] = 2 * h[i]
        matrix[1 + 4 * i][2 + 4 * i] = 1
        matrix[1 + 4 * i][6 + 4 * i] = -1

        matrix[2 + 4 * i][0 + 4 * i] = 6 * h[i]
        matrix[2 + 4 * i][1 + 4 * i] = 2
        matrix[2 + 4 * i][5 + 4 * i] = -2

        matrix[3 + 4 * i][3 + 4 * i] = 1

    for i in range(n-1, n):
        matrix[-4][0+i*4] = h[-1] ** 3
        matrix[-4][1+i*4] = h[-1] ** 2
        matrix[-4][2+i*4] = h[-1]
        matrix[-4][3+i*4] = 1
        matrix[-4][3] = -1

        matrix[-3][0+i*4] = 3 * h[-1] ** 2
        matrix[-3][1+i*4] = 2 * h[-1]
        matrix[-3][2+i*4] = 1
        matrix[-3][2] = -1

        matrix[-2][0+i*4] = 6 * h[-1]
        matrix[-2][1+i*4] = 2
        matrix[-2][1] = -2

        matrix[-1][3+i*4] = 1

    rsm = np.zeros(gr)
    for i in range(n):
        rsm[3+4*i] = y[i]

    lm = np.linalg.solve(matrix, rsm)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        a[i] = lm[0+4*i]
        b[i] = lm[1+4*i]
        c[i] = lm[2+4*i]
        d[i] = lm[3+4*i]

    return a, b, c, d, x_all

# Polynom aufstellen beliebigen Grades
def poly(x,xi,*koeffizienten):
    ergebnis = 0
    grad = len(koeffizienten)-1
    for i , k in enumerate(koeffizienten):
        ergebnis += k *(x-xi)**(grad-i)
    return ergebnis


# kubischer Mittel-Spline (unbrauchbares Ergebnis)
def mittelspline(x,y,m,peri):
    x_all = np.append(x, peri)
    h= np.diff(x_all)
    n = len(x)
    gr = n*4
    matrix = np.zeros((gr, gr))

    for i in range(n - 1):
        # Stützpunkt
        matrix[0 + 4 * i][0 + 4 * i] = h[i] ** 3
        matrix[0 + 4 * i][1 + 4 * i] = h[i] ** 2
        matrix[0 + 4 * i][2 + 4 * i] = h[i]
        matrix[0 + 4 * i][3 + 4 * i] = 1
        matrix[0 + 4 * i][7 + 4 * i] = -1

        # 1. Ableitung
        matrix[1 + 4 * i][0 + 4 * i] = 3 * h[i] ** 2
        matrix[1 + 4 * i][1 + 4 * i] = 2 * h[i]
        matrix[1 + 4 * i][2 + 4 * i] = 1
        matrix[1 + 4 * i][6 + 4 * i] = -1

        # Mittelwert
        matrix[2 + 4 * i][0 + 4 * i] = (1 / 4) * h[i] ** 3
        matrix[2 + 4 * i][1 + 4 * i] = (1 / 3) * h[i] ** 2
        matrix[2 + 4 * i][2 + 4 * i] = (1 / 2) * h[i]
        matrix[2 + 4 * i][3 + 4 * i] = 1

        # Stützpunkt
        matrix[3 + 4 * i][3 + 4 * i] = 1


    for i in range(n-1, n):
        matrix[-4][0+i*4] = h[-1] ** 3
        matrix[-4][1+i*4] = h[-1] ** 2
        matrix[-4][2+i*4] = h[-1]
        matrix[-4][3+i*4] = 1
        matrix[-4][3] = -1

        matrix[-3][0+i*4] = 3 * h[-1] ** 2
        matrix[-3][1+i*4] = 2 * h[-1]
        matrix[-3][2+i*4] = 1
        matrix[-3][2] = -1

        matrix[-2][0+4*i] = (1 / 4) * h[-1] ** 3
        matrix[-2][1+4*i] = (1 / 3) * h[-1] ** 2
        matrix[-2][2+4*i] = (1 / 2) * h[-1]
        matrix[-2][3+4*i] = 1

        matrix[-1][3+i*4] = 1

    smin = min(np.linalg.svd(matrix)[1])
    smax = max(np.linalg.svd(matrix)[1])
    #print(f"Minimum(mittel): {smin}, Maximum(mittel): {smax}")
    #print(f"Konditionszahl(mittel): {smax/smin}")
    #print(f"reziproke Konditionszahl(mittel): {smin/smax}")

    rsm = np.zeros(gr)
    for i in range(n):
        rsm[2 + 4 * i] = m[i]
        rsm[3 + 4 * i] = y[i]

    lm = np.linalg.solve(matrix, rsm)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        a[i] = lm[0+4*i]
        b[i] = lm[1+4*i]
        c[i] = lm[2+4*i]
        d[i] = lm[3+4*i]

    #plt.figure()
    #plt.spy(matrix, markersize=8)  # markersize bestimmt Punktgröße
    #plt.show()

    return a, b, c, d, x_all

# quartischer Mittelspline
def vier_mittel_spline(x,y,peri,m):
    x_all = np.append(x, peri)
    h= np.diff(x_all)
    n = len(x)
    gr = n*5
    matrix = np.zeros((gr, gr))
    # Belegen der Matrix (immer 5 Zeilen außer die letzten 5 Zeilen)
    for i in range(n - 1):
        # Stützpunkt
        matrix[0 + 5 * i][0 + 5 * i] = h[i] ** 4
        matrix[0 + 5 * i][1 + 5 * i] = h[i] ** 3
        matrix[0 + 5 * i][2 + 5 * i] = h[i] ** 2
        matrix[0 + 5 * i][3 + 5 * i] = h[i]
        matrix[0 + 5 * i][4 + 5 * i] = 1
        matrix[0 + 5 * i][9 + 5 * i] = -1

        # Ableitung
        matrix[1 + 5 * i][0 + 5 * i] = 4 * h[i] ** 3
        matrix[1 + 5 * i][1 + 5 * i] = 3 * h[i] ** 2
        matrix[1 + 5 * i][2 + 5 * i] = 2 * h[i]
        matrix[1 + 5 * i][3 + 5 * i] = 1
        matrix[1 + 5 * i][8 + 5 * i] = -1

        # 2. Ableitung
        matrix[2 + 5 * i][0 + 5 * i] = 12 * h[i] ** 2
        matrix[2 + 5 * i][1 + 5 * i] = 6 * h[i]
        matrix[2 + 5 * i][2 + 5 * i] = 2
        matrix[2 + 5 * i][7 + 5 * i] = -2

        # Stützpunkt
        matrix[3 + 5 * i][4 + 5 * i] = 1

        # Mittelwertbedingung
        matrix[4 + 5 * i][0+5*i] = (1 / 5) * h[i] ** 4
        matrix[4 + 5 * i][1+5*i] = (1 / 4) * h[i] ** 3
        matrix[4 + 5 * i][2+5*i] = (1 / 3) * h[i] ** 2
        matrix[4 + 5 * i][3+5*i] = (1 / 2) * h[i] ** 1
        matrix[4 + 5 * i][4+5*i] = 1

    for i in range(n-1, n):
        # Stützpunkt
        matrix[-5][0+i*5] = h[-1] ** 4
        matrix[-5][1+i*5] = h[-1] ** 3
        matrix[-5][2+i*5] = h[-1] ** 2
        matrix[-5][3+i*5] = h[-1]
        matrix[-5][4+i*5] = 1
        matrix[-5][4] = -1

        # Ableitung
        matrix[-4][0+i*5] = 4 * h[-1] ** 3
        matrix[-4][1+i*5] = 3 * h[-1] ** 2
        matrix[-4][2+i*5] = 2 * h[-1]
        matrix[-4][3+i*5] = 1
        matrix[-4][3] = -1

        # 2. Ableitung
        matrix[-3][0+i*5] = 12 * h[-1] ** 2
        matrix[-3][1+i*5] = 6 * h[-1]
        matrix[-3][2+i*5] = 2
        matrix[-3][2] = -2

        # Stützpunkt
        matrix[-2][4+i*5] = 1
        # Mittelwertbedingung
        matrix[-1][0+5*i] = (1 / 5) * h[-1] ** 4
        matrix[-1][1+5*i] = (1 / 4) * h[-1] ** 3
        matrix[-1][2+5*i] = (1 / 3) * h[-1] ** 2
        matrix[-1][3+5*i] = (1 / 2) * h[-1] ** 1
        matrix[-1][4 +5 *i] = 1


    #smin = min(np.linalg.svd(matrix)[1])
    #smax = max(np.linalg.svd(matrix)[1])
    #print(f"Minimum(mittel): {smin}, Maximum(mittel): {smax}")
    #print(f"Konditionszahl(mittel): {smax/smin}")
    #print(f"reziproke Konditionszahl(mittel): {smin/smax}")

    rsm = np.zeros(gr)

    for i in range(n):
        rsm[3+5*i] = y[i]
        rsm[4 + 5 * i] = m[i]

    lm = np.linalg.solve(matrix, rsm)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    e = np.zeros(n)

    for i in range(n):
        a[i] = lm[0+5*i]
        b[i] = lm[1+5*i]
        c[i] = lm[2+5*i]
        d[i] = lm[3+5*i]
        e[i] = lm[4+5*i]

    #print(a)
    #print(b)
    #print(c)
    #print(d)
    #print(e)

    #plt.figure()
    #plt.spy(matrix, markersize=5)  # markersize bestimmt Punktgröße
    #plt.show()

    return a, b, c, d, e, x_all

# quartischer stückweiser Hermite-Mittel-Spline
def hermite_vier_spline(x, y, dy, m):
    n = len(x)

    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    e = np.zeros(n - 1)

    for i in range(n - 1):
        # Belegen der Werte für jedes Intervall,
        # pro Schleifendurchlauf ein Intervall
        xi, xi1 = x[i], x[i + 1]
        yi, yi1 = y[i], y[i + 1]
        dyi, dyi1 = dy[i], dy[i + 1]
        h = xi1 - xi
        mi = m[i]

        # Berechnung der Koeffizienten ohne Gleichungssystem in Abhängigkeit
        # der bekannten Werte der rechten Seite
        a[i] = (30 * (mi - (1 / 2) * (yi1 + yi) + (1 / 12) * h * (dyi1 - dyi))) / h ** 4
        b[i] = (28 * yi1 + 32 * yi -60 * mi - 4 * dyi1 * h + 6 * dyi * h) / h**3
        c[i] = (30 * mi + (3/2) * dyi1 * h -(9/2) * dyi * h - 12 * yi1 - 18 * yi) / h**2
        d[i] = dyi
        e[i] = yi


    x_all = x
    return a, b, c, d, e, x_all
