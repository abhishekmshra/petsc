#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsnes.c,v 1.15 1998/03/06 00:06:09 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "snes.h"

#ifdef HAVE_FORTRAN_CAPS
#define snesregisterdestroy_         SNESREGISTERDESTROY
#define snessetjacobian_             SNESSETJACOBIAN
#define snescreate_                  SNESCREATE
#define snessetfunction_             SNESSETFUNCTION
#define snessetminimizationfunction_ SNESSETMINIMIZATIONFUNCTION
#define snesgetsles_                 SNESGETSLES
#define snessetgradient_             SNESSETGRADIENT
#define snessethessian_              SNESSETHESSIAN
#define snessetmonitor_              SNESSETMONITOR
#define snessetconvergencetest_      SNESSETCONVERGENCETEST
#define snesregisterdestroy_         SNESREGISTERDESTROY
#define snesgetsolution_             SNESGETSOLUTION
#define snesgetsolutionupdate_       SNESGETSOLUTIONUPDATE
#define snesgetfunction_             SNESGETFUNCTION
#define snesgetgradient_             SNESGETGRADIENT
#define snesdestroy_                 SNESDESTROY
#define snesgettype_                 SNESGETTYPE
#define snessetoptionsprefix_        SNESSETOPTIONSPREFIX 
#define snesappendoptionsprefix_     SNESAPPENDOPTIONSPREFIX 
#define snesdefaultmatrixfreematcreate_ SNESDEFAULTMATRIXFREEMATCREATE
#define snessettype_                    SNESSETTYPE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define snesregisterdestroy_         snesregisterdestroy
#define snessetjacobian_             snessetjacobian
#define snescreate_                  snescreate
#define snessetfunction_             snessetfunction
#define snessethessian_              snessethessian
#define snessetgradient_             snessetgradient
#define snesgetsles_                 snesgetsles
#define snessetminimizationfunction_ snessetminimizationfunction
#define snesdestroy_                 snesdestroy
#define snessetmonitor_              snessetmonitor
#define snessetconvergencetest_      snessetconvergencetest
#define snesregisterdestroy_         snesregisterdestroy
#define snesgetsolution_             snesgetsolution
#define snesgetsolutionupdate_       snesgetsolutionupdate
#define snesgetfunction_             snesgetfunction
#define snesgetgradient_             snesgetgradient
#define snesgettype_                 snesgettype
#define snessetoptionsprefix_        snessetoptionsprefix 
#define snesappendoptionsprefix_     snesappendoptionsprefix
#define snesdefaultmatrixfreematcreate_ snesdefaultmatrixfreematcreate
#define snessettype_                    snessettype
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void snessettype_(SNES snes,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = SNESSetType((SNES)PetscToPointer(snes),t);
  FREECHAR(itmethod,t);
}

void snesappendoptionsprefix_(SNES snes,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESAppendOptionsPrefix((SNES)PetscToPointer(snes),t);
  FREECHAR(prefix,t);
}

void snesdefaultmatrixfreematcreate_(SNES snes,Vec x,Mat *J, int *__ierr ){
  Mat mm;
  *__ierr = SNESDefaultMatrixFreeMatCreate(
	(SNES)PetscToPointer(snes),
	(Vec)PetscToPointer(x),&mm);
  *(PetscFortranAddr*) J = PetscFromPointer(mm);
}

static int (*f7)(PetscFortranAddr*,int*,double*,void*,int*);
static int oursnesmonitor(SNES snes,int i,double d,void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1;

  s1 = PetscFromPointer(snes);
  (*f7)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  return 0;
}
void snessetmonitor_(SNES snes,int (*func)(PetscFortranAddr*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = SNESSetMonitor(
	(SNES)PetscToPointer(snes),oursnesmonitor,mctx);
}
static int (*f8)(PetscFortranAddr*,double*,double*,double*,void*,int*);
static int oursnestest(SNES snes,double a,double d,double c,void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1;

  s1 = PetscFromPointer(snes);
  (*f8)(&s1,&a,&d,&c,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  return 0;
}
void snessetconvergencetest_(SNES snes,
       int (*func)(PetscFortranAddr*,double*,double*,double*,void*,int*),
       void *cctx, int *__ierr ){
  f8 = func;
  *__ierr = SNESSetConvergenceTest(
	(SNES)PetscToPointer(snes),oursnestest,cctx);
}
void snesgetsolution_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolution((SNES)PetscToPointer(snes),&rr);
  *(PetscFortranAddr*) x = PetscFromPointer(rr);  
}
void snesgetsolutionupdate_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolutionUpdate((SNES)PetscToPointer(snes),&rr);
  *(PetscFortranAddr*) x = PetscFromPointer(rr);  
}
void snesgetfunction_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetFunction((SNES)PetscToPointer(snes),&rr);
  *(PetscFortranAddr*) r = PetscFromPointer(rr);
}
void snesgetgradient_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetGradient((SNES)PetscToPointer(snes),&rr);
  *(PetscFortranAddr*) r = PetscFromPointer(rr);
}

void snesdestroy_(SNES snes, int *__ierr ){
  *__ierr = SNESDestroy((SNES)PetscToPointer(snes));
  PetscRmPointer(snes);
}
void snesgetsles_(SNES snes,SLES *sles, int *__ierr ){
  SLES s;
  *__ierr = SNESGetSLES((SNES)PetscToPointer(snes),&s);
  *(PetscFortranAddr*) sles = PetscFromPointer(s);
}
static int (*f6)(PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,int*,void*,int*);
static int oursneshessianfunction(SNES snes,Vec x,Mat* mat,Mat* pmat,
                                  MatStructure* st,void *ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3,s4,o3,o4;

  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  o3 = s3 = PetscFromPointer(*mat);
  o4 = s4 = PetscFromPointer(*pmat);
  (*f6)(&s1,&s2,&s3,&s4,(int*)st,ctx,&ierr); CHKERRQ(ierr);
  if (o3 != s3) *mat  = (Mat) PetscToPointer(&s3);
  if (o4 != s4) *pmat = (Mat) PetscToPointer(&s4);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&o3);
  PetscRmPointer(&o3);
  return 0;
}
void snessethessian_(SNES snes,Mat A,Mat B,int (*func)(PetscFortranAddr*,PetscFortranAddr*,
                 PetscFortranAddr*,PetscFortranAddr*,int*,void*,int*),void *ctx, int *__ierr ){
  f6 = func;
  *__ierr = SNESSetHessian(
	(SNES)PetscToPointer(snes),
	(Mat)PetscToPointer(A),
	(Mat)PetscToPointer(B),oursneshessianfunction,ctx);
}

static int (*f5)(PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,void*,int*);
static int oursnesgradientfunction(SNES snes,Vec x,Vec d,void *ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3;

  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(d);
  (*f5)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&s3);
  return 0;
}
void snessetgradient_(SNES snes,Vec r,int (*func)(PetscFortranAddr*,PetscFortranAddr*,
               PetscFortranAddr*,void*,int*),void *ctx, int *__ierr ){
  f5 = func;
  *__ierr = SNESSetGradient(
	(SNES)PetscToPointer(snes),
	(Vec)PetscToPointer(r),oursnesgradientfunction,ctx);
}

static int (*f4)(PetscFortranAddr*,PetscFortranAddr*,double*,void*,int*);
static int oursnesminfunction(SNES snes,Vec x,double* d,void *ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2;

  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  (*f4)(&s1,&s2,d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  return 0;
}
void snessetminimizationfunction_(SNES snes,
          int (*func)(PetscFortranAddr*,PetscFortranAddr*,double*,void*,int*),void *ctx, int *__ierr ){
  f4 = func;
  *__ierr = SNESSetMinimizationFunction(
	(SNES)PetscToPointer(snes),oursnesminfunction,ctx);
}

static int (*f2)(PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,void*,int*);
static int oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3;

  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(f);
  (*f2)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&s3);
  return 0;
}
void snessetfunction_(SNES snes,Vec r,int (*func)(PetscFortranAddr*,PetscFortranAddr*,
       PetscFortranAddr*,void*,int*),void *ctx, int *__ierr ){
   f2 = func;
   *__ierr = SNESSetFunction(
	(SNES)PetscToPointer(snes),
	(Vec)PetscToPointer(r),oursnesfunction,ctx);
}
/* ---------------------------------------------------------*/
void snescreate_(MPI_Comm *comm,SNESProblemType *type,SNES *outsnes, int *__ierr ){
  SNES snes;
*__ierr = SNESCreate(
	(MPI_Comm)PetscToPointerComm( *comm ),*type,&snes);
  *(PetscFortranAddr*)outsnes = PetscFromPointer(snes);
}
/* ---------------------------------------------------------*/
static int (*f3)(PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,MatStructure*,void*,int*);
static int oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,
                          void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3,s4,s3_o,s4_o;

  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f3)(&s1,&s2,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(&s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(&s4);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&s3);
  PetscRmPointer(&s4);
  return 0;
}

void snessetjacobian_(SNES snes,Mat A,Mat B,int (*func)(PetscFortranAddr*,PetscFortranAddr*,
             PetscFortranAddr*,PetscFortranAddr*,MatStructure*,void*,int*),void *ctx, int *__ierr )
{
  f3 = func;
  *__ierr = SNESSetJacobian(
	(SNES)PetscToPointer(snes),
	(Mat)PetscToPointer(A),
	(Mat)PetscToPointer(B),oursnesjacobian,ctx);
}

/* -------------------------------------------------------------*/

void snesregisterdestroy_(int *__ierr)
{
  *__ierr = SNESRegisterDestroy();
}

void snesgettype_(SNES snes,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = SNESGetType((SNES)PetscToPointer(snes),&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

#if defined(__cplusplus)
}
#endif

