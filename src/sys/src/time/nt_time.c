/*$Id: nt_time.c,v 1.20 2000/04/09 04:34:46 bsmith Exp bsmith $*/

#include <petsc.h>
#if defined (PARCH_win32_gnu) || defined (PARCH_win32)
#include <Windows.h>
#define FACTOR   4294967296.0 /* pow(2,32) */

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"nt_time"
PLogDouble nt_time(void) 
{
  static PetscTruth flag = PETSC_TRUE;
  int               ierr;

  static LARGE_INTEGER StartTime,PerfFreq,CurTime; 
  static PLogDouble SecInTick=0.0;
  
  DWORD      dwStartHigh,dwCurHigh;
  PLogDouble dTime,dHigh;
  PLogDouble time;
  
  
  PetscFunctionBegin;
  if (flag) {
    ierr = QueryPerformanceCounter(&StartTime);CHKERRQ(!ierr);
    ierr = QueryPerformanceFrequency(&PerfFreq);CHKERRQ(!ierr);
    /* Explicitly convert the higher 32 bits, and add the lower 32 bits from the counter */
    /* works on non-pentium CPUs ? */
    SecInTick = 1.0/((double)PerfFreq.HighPart*FACTOR+(double)PerfFreq.LowPart);
    flag = PETSC_FALSE;
  }		
  
  ierr        = QueryPerformanceCounter(&CurTime);CHKERRQ(!ierr);
  dwCurHigh   = (DWORD)CurTime.HighPart;
  dwStartHigh = (DWORD)StartTime.HighPart;
  dHigh       = (signed)(dwCurHigh - dwStartHigh);

  dTime = dHigh*(double)FACTOR + (double)CurTime.LowPart - (double)StartTime.LowPart;
  /* Use the following with older versions of the Borland compiler
  dTime = dHigh*(double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
  */
  time  = (double)SecInTick*dTime;

  PetscFunctionReturn(time);
}
#endif
