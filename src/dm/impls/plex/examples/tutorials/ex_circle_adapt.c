static char help[] = "Adaptively refine a box mesh on the boundary of a circle\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscmath.h>

static PetscErrorCode AdaptiveCircumferenceRefinement(DM cdm, Vec coordinates, PetscInt p, PetscInt xcenter, PetscInt ycenter, PetscInt cstart, PetscInt cend, DMLabel *adaptLabel)
{
  PetscInt c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (c = cstart; c < cend; ++c)
    {
      PetscInt       csize;
      PetscScalar    *coords = NULL;
      PetscReal	     min, x1, x2, x3, x4, y1, y2, y3, y4, diag, S1, S2, S3, S4;

      ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);

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
        ierr = DMLabelSetValue(*adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);
      }
      else
      {
        ierr = DMLabelSetValue(*adaptLabel,c,DM_ADAPT_KEEP);CHKERRQ(ierr);
      }

      ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
    }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, cdm, base, preforest, postforest;
  Vec            coordinates;
  PetscInt	 dim = 2, cstart, cend, n, xcenter, ycenter, p = 1, nrefine = 4;
  PetscReal      lower[2] = {0,0}, upper[2] = {4,4};
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel        adaptLabel = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-nrefine", &nrefine, NULL);CHKERRQ(ierr);

  xcenter = (upper[0]+lower[0])/2;
  ycenter = (upper[1]+lower[1])/2;

  /* Create a base mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, lower, upper, NULL, interpolate, &base);CHKERRQ(ierr);

  ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD, &preforest);CHKERRQ(ierr);
  ierr = DMSetType(preforest, (dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preforest, base);CHKERRQ(ierr);
  ierr = DMSetUp(preforest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);


  for (n = 0; n < nrefine; ++n)
  {

    ierr = DMForestGetCellChart(preforest, &cstart, &cend);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(preforest, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(preforest, &coordinates);CHKERRQ(ierr);
    ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);

    ierr = AdaptiveCircumferenceRefinement(cdm, coordinates, p, xcenter, ycenter, cstart, cend, &adaptLabel);CHKERRQ(ierr);
    ierr = DMForestTemplate(preforest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
    ierr = DMDestroy(&preforest);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityLabel(postforest,adaptLabel);CHKERRQ(ierr);
    ierr = DMSetUp(postforest);CHKERRQ(ierr);

    ierr = DMForestGetCellChart(postforest, &cstart, &cend);CHKERRQ(ierr);

    ierr = DMForestTemplate(postforest, PETSC_COMM_WORLD, &preforest);CHKERRQ(ierr);
    ierr = DMDestroy(&postforest);CHKERRQ(ierr);
    ierr = DMSetUp(preforest);CHKERRQ(ierr);
    ierr = DMLabelReset(adaptLabel);CHKERRQ(ierr);
  }

  ierr = DMConvert(preforest, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&preforest);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Vector View*/
/*
  Vec            u;
  PetscViewer    viewer;

  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "sol.vtk");CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
*/

  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

