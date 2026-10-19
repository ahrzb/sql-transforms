#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
int main(void){
  /* every 2^-26 step of m in [1,2): 2^26 values */
  long n=1L<<26, diff=0; double vals[64]; int nv=0;
  for(long i=0;i<n;i+=8){
    double m[8], r[8];
    for(int k=0;k<8;k++) m[k]=1.0+(double)(i+k)/(double)n;
    __m512d v=_mm512_loadu_pd(m);
    __m512d rc=_mm512_roundscale_pd(_mm512_rcp14_pd(v), 0x58);
    _mm512_storeu_pd(r, rc);
    for(int k=0;k<8;k++){
      /* exact-ish RN of 32/m to integer (ties even); 32/m in (16,32], m has 27 bits so 32/m exact enough in long double */
      long double q = 32.0L/(long double)m[k];
      long double fl = floorl(q), fr = q-fl; long double rn;
      if (fr>0.5L) rn=fl+1; else if (fr<0.5L) rn=fl; else rn = (fmodl(fl,2.0L)==0)?fl:fl+1;
      if ((double)(rn/32.0L)!=r[k]) diff++;
      int f=0; for(int t=0;t<nv;t++) if(vals[t]==r[k]) f=1;
      if(!f && nv<64) vals[nv++]=r[k];
    }
  }
  printf("distinct=%d diff=%ld frac=%.3e\n", nv, diff, (double)diff/n);
  return 0;
}
