/*
 * lib/maj3.h — Majority-of-3 Bit-Logik (header only)
 * ==================================================
 *
 * Stellt bereit:
 *   maj3(a,b,c)              — Majority von 3 uint32 (Bit-parallel)
 *   majority_tree(vals, n)  — Majority-Tree über n Werte
 *
 * Fix 2026-06-17: Passthrough-Kaskade bei n = 3^k+1
 *   Bei n%3==1 in JEDER Stufe wurde ein einzelner Wert ohne Majority-Vote
 *   durch alle Stufen getragen → letzte Stufe: AND blockiert Ergebnis.
 *   Fix: n%3==1 → Verarbeite n-4 als 3er-Gruppen + 4er-Gruppe.
 *
 * Nutzung:
 *   #include "maj3.h"
 *   uint32_t r = majority_tree(data, 196);
 *
 * BUF: Puffergröße (Anzahl uint32), default 4096 = 16 KB Stack
 *      Überschreibbar via -DMAJ3_BUF=8192
 */
#ifndef MAJ3_H
#define MAJ3_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef MAJ3_BUF
#  define MAJ3_BUF  4096
#endif


/**
 * maj3(a,b,c) — Bitweise Majority von 3 uint32.
 *
 * Für jedes Bit: 1 wenn mindestens 2 der 3 Eingänge 1 sind.
 *   result = (a & b) | (a & c) | (b & c)
 */
static inline uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) | (a & c) | (b & c);
}


/**
 * majority_tree(vals, n) — Majority-Tree über n uint32-Werte.
 *
 * Reduziert n uint32-Werte auf 1 uint32 via majority-Baum.
 * Jede Stufe: partitioniere in 3er-Gruppen → maj3 pro Gruppe.
 *
 * Fix für n%3==1 (2026-06-17):
 *   Original: letztes Element als Passthrough → bei n=3^k+1 Kaskade
 *   → ein Wert blockiert finales AND → Ergebnis ≈ 0.
 *   Neu: verarbeite n-4 als 3er-Gruppen + letzte 4 als 4er-Gruppe
 *   → majority of 4 = (a&b&c)|(a&b&d)|(a&c&d)|(b&c&d)
 *
 * @param vals  Zeiger auf n uint32-Werte
 * @param n     Anzahl Werte (> 0, ≤ MAJ3_BUF bei Stack-Nutzung)
 * @return      Majority über alle n Werte (pro Bit)
 */
static inline uint32_t majority_tree(const uint32_t *vals, int n) {
    uint32_t buf[MAJ3_BUF];
    if (n <= 0) return 0;
    if (n > MAJ3_BUF) n = MAJ3_BUF;
    memcpy(buf, vals, (size_t)n * sizeof(uint32_t));

    while (n > 1) {
        int n_out = 0;

        if (n % 3 == 1 && n > 4) {
            /* Fix für n%3==1: 4er-Gruppe am Ende statt Passthrough-Kaskade */
            int full = (n - 4) / 3;
            for (int i = 0; i < full * 3; i += 3)
                buf[n_out++] = maj3(buf[i], buf[i+1], buf[i+2]);
            int j = full * 3;
            uint32_t a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3];
            /* majority of 4: mindestens 3 von 4 müssen 1 sein */
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d);

        } else if (n % 3 == 2 && n > 5) {
            /* Fix für n%3==2: 5er-Gruppe am Ende statt AND-Kaskade */
            int full = (n - 5) / 3;
            for (int i = 0; i < full * 3; i += 3)
                buf[n_out++] = maj3(buf[i], buf[i+1], buf[i+2]);
            int j = full * 3;
            uint32_t a = buf[j], b = buf[j+1], c = buf[j+2], d = buf[j+3], e = buf[j+4];
            /* majority of 5: mindestens 3 von 5 müssen 1 sein */
            buf[n_out++] = (a & b & c) | (a & b & d) | (a & b & e)
                         | (a & c & d) | (a & c & e) | (a & d & e)
                         | (b & c & d) | (b & c & e) | (b & d & e)
                         | (c & d & e);

        } else {
            /* Normaler Pfad: 3er-Gruppen + Rest */
            for (int i = 0; i < n; i += 3) {
                uint32_t a = buf[i];
                if (i + 2 < n)
                    buf[n_out++] = maj3(a, buf[i+1], buf[i+2]);
                else if (i + 1 < n)
                    buf[n_out++] = a & buf[i+1];
                else
                    buf[n_out++] = a;
            }
        }
        n = n_out;
    }
    return buf[0];
}

#endif /* MAJ3_H */
