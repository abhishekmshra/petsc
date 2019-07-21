
static char help[] = "Tests ISOnComm.\n\n";

#include <petscis.h>
#include <petscviewer.h>

static PetscErrorCode CreateSeqIS(MPI_Comm comm, PetscInt type, PetscInt bs, PetscInt n, PetscInt first, PetscInt step, PetscInt omit, IS *is)
{
  PetscInt       *idx, i, j;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  *is = NULL;
  if (omit && !(rank % omit)) PetscFunctionReturn(0);
  first += 100*rank;
  if (type == 2) {
    ierr = ISCreateStride(PETSC_COMM_SELF,n,first,step,is);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0,j=first; i<n; i++,j+=step) idx[i] = j;
  if (type == 3) {
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,n,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
    ierr = ISSetBlockSize(*is,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  IS             isseq,is;
  PetscCopyMode  mode=PETSC_USE_POINTER;
  PetscInt       bs=0, n=10, first=0, step=0, type=1, omit=0;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(NULL,NULL,"-mode",PetscCopyModes,(PetscEnum*)&mode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-type",&type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-first",&first,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-omit",&omit,NULL);CHKERRQ(ierr);

  ierr = CreateSeqIS(comm, type, bs, n, first, step, omit, &isseq);CHKERRQ(ierr);
  ierr = ISOnComm(isseq, comm, mode, &is);CHKERRQ(ierr);
  ierr = ISView(is,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);

  ierr = ISDestroy(&isseq);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   testset:
      args: -n 8 -bs 4
      args: -type {{1 2 3}separate output}
      args: -mode {{COPY_VALUES USE_POINTER}}
      test:
        suffix: 1
        nsize: 1
        args: -first 0
        args: -step 1
      test:
        suffix: 4
        nsize: 4
        args: -first {{-5 5}separate output}
        args: -step {{-2 2}separate output}
        args: -omit {{0 2}separate output}

TEST*/
