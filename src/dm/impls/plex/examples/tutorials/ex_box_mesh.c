static char help[] = "Creating a box mesh and refining\n\n";

#include <petscdmplex.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>

int main(int argc, char **argv)
{
  DM             dm, dmDist = NULL;
  Vec            u;
  PetscViewer    viewer;
  PetscInt	 dim = 2;
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel	 adaptLabel = NULL;
//  VecTagger      refineTag = NULL;
//  VecTaggerBox   refineBox;
//  IS		 refineIS;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_TRUE, NULL, NULL, NULL, NULL, interpolate, &dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist;}
  /* Create a Vec with this layout and view it */
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetValue(adaptLabel,2,DM_ADAPT_REFINE);CHKERRQ(ierr);
  ierr = DMAddLabel(dm,adaptLabel);CHKERRQ(ierr);
  ierr = DMAdaptLabel(dm, adaptLabel, &dm);CHKERRQ(ierr);
//  ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
//  ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);

//  ierr = VecTaggerCreate(PETSC_COMM_WORLD,&refineTag);CHKERRQ(ierr);
//  ierr = PetscObjectSetOptionsPrefix((PetscObject)refineTag,"refine_");CHKERRQ(ierr);
//  ierr = VecTaggerSetType(refineTag,VECTAGGERABSOLUTE);CHKERRQ(ierr);
//  ierr = VecTaggerAbsoluteSetBox(refineTag,&refineBox);CHKERRQ(ierr);
//  ierr = VecTaggerSetFromOptions(refineTag);CHKERRQ(ierr);
//  ierr = VecTaggerSetUp(refineTag);CHKERRQ(ierr);
//  ierr = PetscObjectViewFromOptions((PetscObject)refineTag,NULL,"-tag_view");CHKERRQ(ierr);
  

//  ierr = VecTaggerComputeIS(refineTag,u,&refineIS);CHKERRQ(ierr);
//  ierr = ISGetSize(refineIS,&nRefine);CHKERRQ(ierr);
//  ierr = DMLabelSetStratumIS(adaptLabel,DM_ADAPT_REFINE,refineIS);CHKERRQ(ierr);
//  ierr =  ISDestroy(&refineIS);CHKERRQ(ierr);


//  ierr = DMAdaptLabel(dm, adaptLabel, &dm);CHKERRQ(ierr);



  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
//  ierr = PetscPrintf(PETSC_COMM_WORLD, "value: %D",value);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

