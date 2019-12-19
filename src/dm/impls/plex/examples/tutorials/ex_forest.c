static char help[] = "Creating a box mesh and refining\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>

int main(int argc, char **argv)
{
  DM             dm, dmf, dmp;
  Vec            u;
  PetscViewer    viewer;
  PetscInt	 nRefine, dim = 2;
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel	 adaptLabel = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMFOREST);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, interpolate, &dmp);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(dm, dmp);CHKERRQ(ierr);
  ierr = DMForestTemplate(dm, PETSC_COMM_WORLD, &dmf);

  /* Create a Vec with this layout and view it */
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = DMAddLabel(dm,adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetValue(adaptLabel,1,DM_ADAPT_REFINE);CHKERRQ(ierr);
//  ierr = DMLabelGetValue(adaptLabel,1,&value);CHKERRQ(ierr);
//  ierr = DMAdaptLabel(dm, adaptLabel, &dm);CHKERRQ(ierr);
  //ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
//  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(dmf,adaptLabel);CHKERRQ(ierr);
  ierr = DMSetUp(dmf);CHKERRQ(ierr);


//  ierr = DMAdaptLabel(dm, adaptLabel, &dm);CHKERRQ(ierr);


//  ierr = DMConvert(dmf, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
//  ierr = VecView(u, viewer);CHKERRQ(ierr);
//  ierr = PetscPrintf(PETSC_COMM_WORLD, "nRefine: %D",nRefine);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

