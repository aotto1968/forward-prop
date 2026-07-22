/* lib/ki-omp.h — OpenMP portability stubs
 * =========================================
 * Provides fallbacks for OpenMP functions when compiled without -fopenmp.
 * Include this AFTER <omp.h> or let it provide its own stubs.
 */
#ifndef KI_OMP_H
#define KI_OMP_H

#ifdef _OPENMP
#  include <omp.h>
#else
/* Provide stubs for single-threaded execution */
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

#endif /* KI_OMP_H */
