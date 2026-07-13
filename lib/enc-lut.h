/*
 * lib/enc-lut.h — Precomputed Encoding Lookup Table (Legacy Wrapper)
 * ===================================================================
 *
 * This is a legacy wrapper. New users should include
 * "ki-encoding.h" directly.
 *
 * This header includes ki-encoding.h and defines
 * enc_lut_init_all() as a convenience function for the Otto Score.
 *
 * Usage (legacy):
 *   1. #include "ki-common.h" first (defines aa, ki_Args)
 *   2. #include "enc-lut.h"
 *   3. enc_lut_init_all()  — initializes all tables active in aa.enc*
 */
#ifndef ENC_LUT_H
#define ENC_LUT_H

#include "ki-encoding.h"  /* enc_lut_init_enc, enc_lut_get, ki_apply_enc_w */

/* ── Init all tables from aa.enc[] and aa.enc_array[] (legacy) ─── */
/* Requires aa (ki_Args) to be visible. Initialisiert nur eindeutige
 * (encoding, width)-Kombinationen aus aa.enc[] und aa.enc_array[]. */
#ifdef __BUILT_WITH_KI_COMMON
static inline void enc_lut_init_all(void) {
    int done[_KI_ENC_NENC][_KI_ENC_NWI];
    memset(done, 0, sizeof(done));
    for (int b = 0; b < COLOR_NB; b++) {
        int e = (int)aa.enc[b];
        int w = (int)aa.enc_width[b];
        if (e < 0 || e >= _KI_ENC_NENC) continue;
        if (w != 8 && w != 16 && w != 32) w = 8;
        int wi = _enc_lut_wi(w);
        if (done[e][wi]) continue;
        done[e][wi] = 1;
        enc_lut_init_enc(e, w);
    }
    for (int i = 0; i < aa.enc_count; i++) {
        int e = (int)aa.enc_array[i].type;
        int w = (int)aa.enc_array[i].width;
        if (e < 0 || e >= _KI_ENC_NENC) continue;
        if (w != 8 && w != 16 && w != 32) w = 8;
        int wi = _enc_lut_wi(w);
        if (done[e][wi]) continue;
        done[e][wi] = 1;
        enc_lut_init_enc(e, w);
    }
}
#endif /* __BUILT_WITH_KI_COMMON */

#endif /* ENC_LUT_H */
