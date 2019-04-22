#ifndef _SQUARE_DGEMM_H
#define _SQUARE_DGEMM_H
extern void square_dgemm(int N, double A[N][N], double B[N][N], double C[N][N]);
extern const char *dgemm_desc;

//#define BI 192
//#define BJ 192
//#define BK 192


#endif /* _SQUARE_DGEMM_H */

