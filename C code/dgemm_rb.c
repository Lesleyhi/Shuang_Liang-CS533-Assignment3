#include "dgemm.h"
#include <stdio.h>

#include <immintrin.h>
#include <emmintrin.h>
#include "bi.h"

#define BI 32
#define BJ 32
#define BK 32

/* Cache blocking parameters */
#ifndef BI
#define BI _BI
#endif

#ifndef BJ
#define BJ _BI
#endif

#ifndef BK
#define BK _BI
#endif

/* Register blocking parameters */
#ifndef RI
#define RI 2
#endif

#ifndef RJ
#define RJ 2
#endif

#define xstr(s) str(s)
#define str(s) #s

#define CBLOCK_DESCR BI * BJ * BK
#define RBLOCK_DESCR RI * RJ

static inline int min(int i, int j)
{
	return i < j ? i : j;
}

const char * dgemm_desc = "DGEMM cache blocked " xstr(CBLOCK_DESCR) " and register blocked " xstr(RBLOCK_DESCR);

// 用寄存器缓存来计算矩阵乘，矩阵A的大小是 RI × bk, 矩阵B的大小是 bk × RJ，结果C矩阵是 RI x RJ 
void dgebb_subblock_opt(int bk,
			int Astride, double A[][Astride],  
	  	        int Bstride, double B[][Bstride], 
	 	        int Cstride, double C[][Cstride])
{
	double a, blocal[RJ], clocal[RI][RJ];
	int i, j, k;

	// 记住C的值到clocal，clocal应该是register 缓存的东西，因为重复访问的次数最多
	for (i = 0; i < RI; i++)
		for (j = 0; j < RJ; j++)
			clocal[i][j] = C[i][j];

	for (k = 0; k < bk; k++) {
		// 把B的数也要放到blocal中去，但是重复使用的次数会比cloacl少
		for (j = 0; j < RJ; j++) {
 			blocal[j] = B[k][j];
		}
		for (i = 0; i < RI; i++) {
			a = A[i][k];
			for (j = 0; j < RJ; j++) {
				clocal[i][j] = clocal[i][j] + a * blocal[j];
			}
		}
	}

	// 计算完成以后，把clocal的内容再放回到C中去
	for (i = 0; i < RI; i++) {
		for (j = 0; j < RJ; j++) {
			C[i][j] = clocal[i][j];
		}
	}
}

void dgebb_subblock_gen(int bi, int bj, int bk,
			int Astride, double A[][Astride],  
			int Bstride, double B[][Bstride], 
			int Cstride, double C[][Cstride])
{
	int i, k, j;
	for (i = 0; i < bi; i++) {
		for (k = 0; k < bk; k++) {
			double tmp = A[i][k];
			for (j = 0; j < bj; j++) {
				C[i][j] += tmp * B[k][j];
			}
		}
	}
}

/* Multiply a bi*bk block of A times a bk*bj block of B 
 * into an bi * bj block of C. Subblock for registers when possible */
// 这个也是一个矩阵乘法，它会切分成更加小的block（RI x RJ 大小） 用寄存器缓存优化算法来计算
void dgebb(int bi, int bj, int bk,
	   int Astride, double A[][Astride],  
	   int Bstride, double B[][Bstride], 
	   int Cstride, double C[][Cstride])
{

        int i, j;

        /* Get the main body of the block */
        for (i = 0; i < bi; i+=RI) {
                int bii = min(RI, bi - i);
                for (j = 0; j < bj; j+= RJ) {
                        int bjj = min(RJ, bj - j);
                        if ((bii == RI) && (bjj == RJ)) {
                                dgebb_subblock_opt(bk,
                                                Astride, (double (*)[])&A[i][0],
                                                Bstride, (double (*)[])&B[0][j],
                                                Cstride, (double (*)[])&C[i][j]);
                        } else {
                                dgebb_subblock_gen(bii, bjj, bk,
                                                Astride, (double (*)[])&A[i][0],
                                                Bstride, (double (*)[])&B[0][j],
                                                Cstride, (double (*)[])&C[i][j]);
                        }
                }
        }
}

/* Multiply a NxBK panel of A times a BKxBJ block of B
 * into a NxBJ panel of C. */
void dgepb(int N, int bj, int bk, 
	   int Astride, double A[N][Astride],  
	   int Bstride, double B[N][Bstride], 
	   int Cstride, double C[N][Cstride])
{
	int i;

	for (i = 0; i < N; i += BI) {
		int bi = min(BI, N - i);
		dgebb(bi, bj, bk, 
		      Astride, (double (*)[])&A[i][0],
		      Bstride, B,
		      Cstride, (double (*)[])&C[i][0]);
	}
}

/* Breaks dgemm down into panels of A and C and blocks of B */
void square_dgemm(int N, double A[N][N], double B[N][N], double C[N][N])
{
	int j, k;
	for (k = 0; k < N; k += BK) {
		int bk = min(BK, N - k);
		for (j = 0; j < N; j += BJ)  {
			int bj = min(BJ, N - j);
			dgepb(N, bj, bk,
			      N, (double (*)[N])&A[0][k], 
			      N, (double (*)[N])&B[k][j],
			      N, (double (*)[N])&C[0][j]);
		}
	}
}
