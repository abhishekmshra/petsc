static char help[] = "Creating a box mesh and iterating to adaptively refining the centroid using p4est\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>

int main(int argc, char **argv)
{
  DM             dm, cdm, base, preforest, postforest;
  Vec            coordinates;
  PetscInt	 dim = 2, c, csize, cstart, cend, n, nrefine = 2;
  PetscReal      lower[2] = {-1,-1}, upper[2] = {1,1};
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel        adaptLabel = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-nrefine", &nrefine, NULL);CHKERRQ(ierr);

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

    for (c = cstart; c < cend; ++c)
    {
      PetscScalar    *coords = NULL;

      ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
      //ierr = PetscPrintf(PETSC_COMM_WORLD, "x coord: %f y coord: %f\n",PetscRealPart(coords[0]), PetscRealPart(coords[1]));CHKERRQ(ierr);


      if (PetscRealPart(coords[0]) == 0 && PetscRealPart(coords[1]) == 0)
      {
        ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);
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

    ierr = DMForestGetCellChart(postforest, &cstart, &cend);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Start Cell: %D, End Cell: %D\n",cstart,cend);CHKERRQ(ierr);

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

