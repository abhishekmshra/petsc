/*
    Provides access to system related and general utility routines.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode  PetscGetArchType(char[],size_t);
EXTERN PetscErrorCode  PetscGetHostName(char[],size_t);
EXTERN PetscErrorCode  PetscGetUserName(char[],size_t);
EXTERN PetscErrorCode  PetscGetProgramName(char[],size_t);
EXTERN PetscErrorCode  PetscSetProgramName(const char[]);
EXTERN PetscErrorCode  PetscGetDate(char[],size_t);
EXTERN PetscErrorCode  PetscSetInitialDate(void);
EXTERN PetscErrorCode  PetscGetInitialDate(char[],size_t);

EXTERN PetscErrorCode  PetscSortInt(PetscInt,PetscInt[]);
EXTERN PetscErrorCode  PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortReal(PetscInt,PetscReal[]);
EXTERN PetscErrorCode  PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);

EXTERN PetscErrorCode  PetscSetDisplay(void);
EXTERN PetscErrorCode  PetscGetDisplay(char[],size_t);

extern PetscCookie PETSC_RANDOM_COOKIE;

typedef enum { RANDOM_DEFAULT,RANDOM_DEFAULT_REAL,
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue()
S*/
typedef struct _p_PetscRandom*   PetscRandom;

EXTERN PetscErrorCode PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
EXTERN PetscErrorCode PetscRandomGetValue(PetscRandom,PetscScalar*);
EXTERN PetscErrorCode PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
EXTERN PetscErrorCode PetscRandomDestroy(PetscRandom);

EXTERN PetscErrorCode PetscGetFullPath(const char[],char[],size_t);
EXTERN PetscErrorCode PetscGetRelativePath(const char[],char[],size_t);
EXTERN PetscErrorCode PetscGetWorkingDirectory(char[],size_t);
EXTERN PetscErrorCode PetscGetRealPath(char[],char[]);
EXTERN PetscErrorCode PetscGetHomeDirectory(char[],size_t);
EXTERN PetscErrorCode PetscTestFile(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PetscTestDirectory(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PetscBinaryRead(int,void*,int,PetscDataType);
EXTERN PetscErrorCode PetscSynchronizedBinaryRead(MPI_Comm,int,void*,int,PetscDataType);
EXTERN PetscErrorCode PetscBinaryWrite(int,void*,int,PetscDataType,int);
EXTERN PetscErrorCode PetscBinaryOpen(const char[],int,int *);
EXTERN PetscErrorCode PetscBinaryClose(int);
EXTERN PetscErrorCode PetscSharedTmp(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PetscGetTmp(MPI_Comm,char *,int);
EXTERN PetscErrorCode PetscFileRetrieve(MPI_Comm,const char *,char *,int,PetscTruth*);
EXTERN PetscErrorCode PetscLs(MPI_Comm,const char[],char*,int,PetscTruth*);
EXTERN PetscErrorCode PetscDLLibraryCCAAppend(MPI_Comm,PetscDLLibraryList*,const char[]);

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for 
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE    (32/8)
#define PETSC_BINARY_FLOAT_SIZE  (32/8)
#define PETSC_BINARY_CHAR_SIZE    (8/8)
#define PETSC_BINARY_SHORT_SIZE  (16/8)
#define PETSC_BINARY_DOUBLE_SIZE (64/8)
#define PETSC_BINARY_SCALAR_SIZE sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to PetscBinarySeek()

  Level: advanced

.seealso: PetscBinarySeek(), PetscSynchronizedBinarySeek()
E*/
typedef enum {PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1,PETSC_BINARY_SEEK_END = 2} PetscBinarySeekType;
EXTERN PetscErrorCode PetscBinarySeek(int,int,PetscBinarySeekType,int*);
EXTERN PetscErrorCode PetscSynchronizedBinarySeek(MPI_Comm,int,int,PetscBinarySeekType,int*);

EXTERN PetscErrorCode PetscSetDebugger(const char[],PetscTruth);
EXTERN PetscErrorCode PetscSetDefaultDebugger(void);
EXTERN PetscErrorCode PetscSetDebuggerFromString(char*);
EXTERN PetscErrorCode PetscAttachDebugger(void);
EXTERN PetscErrorCode PetscStopForDebugger(void);

EXTERN PetscErrorCode PetscGatherNumberOfMessages(MPI_Comm,int*,int*,int*);
EXTERN PetscErrorCode PetscGatherMessageLengths(MPI_Comm,int,int,int*,int**,int**);
EXTERN PetscErrorCode PetscPostIrecvInt(MPI_Comm,int,int,int*,int*,int***,MPI_Request**);
EXTERN PetscErrorCode PetscPostIrecvScalar(MPI_Comm,int,int,int*,int*,PetscScalar***,MPI_Request**);

EXTERN PetscErrorCode PetscSSEIsEnabled(MPI_Comm,PetscTruth *,PetscTruth *);

/* ParameterDict objects encapsulate arguments to generic functions, like mechanisms over interfaces */
EXTERN PetscErrorCode ParameterDictCreate(MPI_Comm, ParameterDict *);
EXTERN PetscErrorCode ParameterDictDestroy(ParameterDict);
EXTERN PetscErrorCode ParameterDictRemove(ParameterDict, const char []);
EXTERN PetscErrorCode ParameterDictSetInteger(ParameterDict, const char [], int);
EXTERN PetscErrorCode ParameterDictSetDouble(ParameterDict, const char [], double);
EXTERN PetscErrorCode ParameterDictSetObject(ParameterDict, const char [], void *);
EXTERN PetscErrorCode ParameterDictGetInteger(ParameterDict, const char [], int *);
EXTERN PetscErrorCode ParameterDictGetDouble(ParameterDict, const char [], double *);
EXTERN PetscErrorCode ParameterDictGetObject(ParameterDict, const char [], void **);

/* Parallel communication routines */
/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES} InsertMode;

/*M
    INSERT_VALUES - Put a value into a vector or matrix, overwrites any previous value

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*M
    ADD_VALUES - Adds a value into a vector or matrix, if there previously was no value, just puts the
                value into that location

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*M
    MAX_VALUES - Puts the maximum of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: InsertMode, VecScatterBegin(), VecScatterEnd(), ADD_VALUES, INSERT_VALUES

M*/

/*E
  ScatterMode - Determines the direction of a scatter

  Level: beginner

.seealso: VecScatter, VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {SCATTER_FORWARD=0, SCATTER_REVERSE=1, SCATTER_FORWARD_LOCAL=2, SCATTER_REVERSE_LOCAL=3, SCATTER_LOCAL=2} ScatterMode;

/*M
    SCATTER_FORWARD - Scatters the values as dictated by the VecScatterCreate() call

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_REVERSE - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_FORWARD_LOCAL - Scatters the values as dictated by the VecScatterCreate() call except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_REVERSE_LOCAL - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()  except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE

M*/

EXTERN PetscErrorCode PetscGhostExchange(MPI_Comm, int, int *, int *, PetscDataType, int *, InsertMode, ScatterMode, void *, void *);

/* 
  Create and initialize a linked list 
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLCreate(idx_start,lnk_max,nlnk,lnk,bt) 0;\
{\
  PetscMalloc(nlnk*sizeof(int),&lnk);\
  PetscBTCreate(nlnk,bt);\
  ierr = PetscBTMemzero(nlnk,bt);\
  lnk[idx_start] = lnk_max;\
}

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAdd(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  int _k,_entry,_location,_lnkdata;\
  nlnk = 0;\
  _k=nidx;\
  while (_k){/* assume indices are almost in increasing order, starting from its end saves computation */\
    _entry = indices[--_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk if it is new */\
      if (_entry <  _lnkdata){/* new entry */\
        lnk[_location] = _entry;\
        lnk[_entry]    = _lnkdata;\
        nlnk++;\
      }\
    }\
  }\
}
/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices   - array that contains the copied data
    lnk       -llinked list that is cleaned and initialize
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLClean(idx_start,lnk_max,nlnk,lnk,indices,bt) 0;\
{\
  int _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    PetscBTClear(bt,_idx);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscLLDestroy(lnk,bt) 0;\
{\
  PetscFree(lnk);\
  PetscBTDestroy(bt);\
}

PETSC_EXTERN_CXX_END
#endif /* __PETSCSYS_H */
