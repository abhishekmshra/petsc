/* $Id: senddense.c,v 1.39 2000/04/09 04:33:46 bsmith Exp bsmith $ */

#include "src/sys/src/viewer/impls/socket/socket.h"

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketPutScalar_Private"></a>*/"ViewerSocketPutScalar_Private" 
/*
   ViewerSocketPutScalar_Private - Passes an Scalar array to a Socket viewer.

  Input Parameters:
.  viewer - obtained from ViewerSocketOpen()
.  m, n - number of rows and columns of array
.  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ
   either
.vb
     MatView(Mat matrix,Viewer viewer)
              or
     VecView(Vec vector,Viewer viewer)
.ve

.keywords: Viewer, Matlab, put, dense, array, vector

.seealso: ViewerSocketOpen(), MatView(), VecView()
*/
int ViewerSocketPutScalar_Private(Viewer viewer,int m,int n,Scalar *array)
{
  Viewer_Socket *vmatlab = (Viewer_Socket*)viewer->data;
  int           ierr,t = vmatlab->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0);CHKERRQ(ierr); 
#if !defined(PETSC_USE_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_SCALAR,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketPutReal_Private"></a>*/"ViewerSocketPutReal_Private" 
/*
   ViewerSocketPutReal_Private - Passes a double (or single) precision array to 
   a Matlab viewer.

  Input Parameters:
.  viewer - obtained from ViewerSocketOpen()
.  m, n - number of rows and columns of array
.  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ
   either
.vb
     MatView(Mat matrix,Viewer viewer)
              or
     VecView(Vec vector,Viewer viewer)
.ve

.keywords: Viewer, Socket, put, dense, array, vector

.seealso: ViewerSocketOpen(), MatView(), VecView()
*/
int ViewerSocketPutReal_Private(Viewer viewer,int m,int n,PetscReal *array)
{
  Viewer_Socket *vmatlab = (Viewer_Socket*)viewer->data;
  int           ierr,t = vmatlab->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0);CHKERRQ(ierr); 
  value = 0;
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_REAL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketPutInt_Private"></a>*/"ViewerSocketPutInt_Private" 
/*
   ViewerSocketPutInt_Private - Passes an integer array to a Socket viewer.

   Input Parameters:
+  viewer - obtained from ViewerSocketOpen()
.  m - number of rows of array
-  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ either
.vb
     MatView(Mat matrix,Viewer viewer)
              or
     VecView(Vec vector,Viewer viewer)
.ve

.keywords: Viewer, Socket, put, dense, array, vector

.seealso: ViewerSocketOpen(), MatView(), VecView()
*/
int ViewerSocketPutInt_Private(Viewer viewer,int m,int *array)
{
  Viewer_Socket *vmatlab = (Viewer_Socket*)viewer->data;
  int           ierr,t = vmatlab->port,type = DENSEINT;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m,PETSC_INT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

