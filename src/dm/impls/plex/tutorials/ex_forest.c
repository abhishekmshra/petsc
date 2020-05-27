static char help[] = "Creating a box mesh and adaptively refining using p4est\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>

int main(int argc, char **argv)
{
  DM             dm, base, preforest, postforest, pf2, pf3;
  Vec            u;
  PetscViewer    viewer;
  PetscInt	 dim = 2;
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel	 adaptLabel = NULL, aL2 = NULL, aL3 = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a base mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, interpolate, &base);CHKERRQ(ierr);
  ierr = DMRefine(base, PETSC_COMM_WORLD, &base);CHKERRQ(ierr);
  ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD, &preforest);CHKERRQ(ierr);
  ierr = DMSetType(preforest, (dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
//  ierr = DMCopyDisc(base,preforest);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preforest, base);CHKERRQ(ierr);
  ierr = DMSetUp(preforest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);

  /* Create a Vec with this layout and view it */
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);

  ierr = DMLabelSetValue(adaptLabel,1,DM_ADAPT_REFINE);CHKERRQ(ierr);
//  ierr = DMLabelSetValue(adaptLabel,15,DM_ADAPT_REFINE);CHKERRQ(ierr);

  ierr = DMForestTemplate(preforest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
  ierr = DMDestroy(&preforest);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(postforest,adaptLabel);CHKERRQ(ierr);
  ierr = DMSetUp(postforest);CHKERRQ(ierr);

  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&aL2);CHKERRQ(ierr);
  ierr = DMLabelSetValue(aL2,4,DM_ADAPT_REFINE);CHKERRQ(ierr);
//  ierr = DMLabelSetValue(aL2,6,DM_ADAPT_REFINE);CHKERRQ(ierr);

  ierr = DMForestTemplate(postforest, PETSC_COMM_WORLD, &pf2);CHKERRQ(ierr);
  ierr = DMDestroy(&postforest);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(pf2,aL2);CHKERRQ(ierr);
  ierr = DMSetUp(pf2);CHKERRQ(ierr);

  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&aL3);CHKERRQ(ierr);
  ierr = DMLabelSetValue(aL3,6,DM_ADAPT_REFINE);CHKERRQ(ierr);
//  ierr = DMLabelSetValue(aL3,6,DM_ADAPT_REFINE);CHKERRQ(ierr);

  ierr = DMForestTemplate(pf2, PETSC_COMM_WORLD, &pf3);CHKERRQ(ierr);
  ierr = DMDestroy(&pf2);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(pf3,aL3);CHKERRQ(ierr);
  ierr = DMSetUp(pf3);CHKERRQ(ierr);

  ierr = DMConvert(pf3, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&pf3);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

