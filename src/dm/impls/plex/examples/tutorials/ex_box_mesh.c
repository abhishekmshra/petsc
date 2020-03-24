static char help[] = "Creating a box mesh and refining\n\n";

#include <petscdmplex.h>
#include <petscdm.h>

int main(int argc, char **argv)
{
  DM             dm, dmDist;
  PetscInt	 dim = 2;
  PetscBool      interpolate = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, interpolate, &dm);CHKERRQ(ierr);
  /* Create a Vec with this layout and view it */
  ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist)
  {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmDist;
  }
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  /*
  Vec            u;
  PetscViewer    viewer;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "value: %D",value);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
   */
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

