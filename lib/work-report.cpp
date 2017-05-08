#include <stdio.h>
#include <math.h>
long __work_counter = 0;

__attribute__((destructor))
void __no_instr_report_profiling(){
  FILE *fp = fopen("/tmp/profiling.txt","w");
  fprintf(fp, "%lu", __work_counter);
  fclose(fp);
  fprintf(stderr, "Reporting work profiling: /tmp/profiling.txt\n");
}

