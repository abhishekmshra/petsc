static char help[] = "Load a mesh and refine\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscmath.h>

int main(int argc, char **argv)
{
  DM             dm, dmDist, cdm, preforest, postforest;
  Vec            coordinates;
  PetscInt	 dim = 2, c, csize, cstart, cend, n, xcenter = 3, ycenter = 3, nrefine = 2;
  PetscReal      p = 0.5;
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel        adaptLabel = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, "mesh_init.h5", interpolate, &dm);CHKERRQ(ierr);
  /* Create a Vec with this layout and view it */
  //ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  //ierr = DMRefine(dm, PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist)
  {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmDist;
  }

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD, &preforest);CHKERRQ(ierr);
  ierr = DMSetType(preforest, (dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preforest, dm);CHKERRQ(ierr);
  ierr = DMSetUp(preforest);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  for (n = 0; n < nrefine; ++n)
  {

    ierr = DMForestGetCellChart(preforest, &cstart, &cend);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(preforest, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(preforest, &coordinates);CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "c start: %d c end: %d\n",cstart,cend);CHKERRQ(ierr);

    for (c = cstart; c < cend; ++c)
    {
      PetscScalar    *coords = NULL;
      //PetscReal	     min, x1, x2, x3, x4, y1, y2, y3, y4, diag, S1, S2, S3, S4;

      ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
      /*
      x1 = PetscRealPart(coords[0]); x2 = PetscRealPart(coords[2]);
      x3 = PetscRealPart(coords[4]); x4 = PetscRealPart(coords[6]);
      y1 = PetscRealPart(coords[1]); y2 = PetscRealPart(coords[3]);
      y3 = PetscRealPart(coords[5]); y4 = PetscRealPart(coords[7]);

      diag = PetscSqrtReal(PetscSqr(x3-x1) + PetscSqr(y3-y1));

      S1 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x1) + PetscSqr(ycenter-y1)) - p);
      S2 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x2) + PetscSqr(ycenter-y2)) - p);
      S3 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x3) + PetscSqr(ycenter-y3)) - p);
      S4 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x4) + PetscSqr(ycenter-y4)) - p);

      min = PetscMin(PetscMin(S1, S2), PetscMin(S3, S4));

      if (min<diag)
      {
        ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);
      }
      else
      {
        ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_KEEP);CHKERRQ(ierr);
      }
       */
      if (PetscRealPart(coords[0]) == 2 && PetscRealPart(coords[1]) == 2)
      {
        ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "c: %d\n",c);CHKERRQ(ierr);
      }
      else
      {
        ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_KEEP);CHKERRQ(ierr);
      }

      ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
    }

    ierr = DMForestTemplate(preforest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
    ierr = DMDestroy(&preforest);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityLabel(postforest,adaptLabel);CHKERRQ(ierr);
    ierr = DMSetUp(postforest);CHKERRQ(ierr);

    ierr = DMForestTemplate(postforest, PETSC_COMM_WORLD, &preforest);CHKERRQ(ierr);
    ierr = DMDestroy(&postforest);CHKERRQ(ierr);
    ierr = DMSetUp(preforest);CHKERRQ(ierr);
    ierr = DMLabelReset(adaptLabel);CHKERRQ(ierr);
  }

  ierr = DMConvert(preforest, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&preforest);CHKERRQ(ierr);

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

