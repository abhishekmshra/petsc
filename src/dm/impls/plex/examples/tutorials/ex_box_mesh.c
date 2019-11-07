static char help[] = "Creating a box mesh\n\n";

#include <petscdmplex.h>
#include<petscdmlabel.h>
int main(int argc, char **argv)
{
  DM             dm, dmDist = NULL;
  Vec            u;
  PetscViewer    viewer;
  PetscInt	 dim = 2;
  PetscBool      interpolate = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, interpolate, &dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist;}
  /* Create a Vec with this layout and view it */
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
//  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
//  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

