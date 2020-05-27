static char help[] = "Adaptively refine mesh using mutiple external refinement functions\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscmath.h>

typedef struct {
  PetscInt       p, xcenter, ycenter, ytop, xright;
  PetscErrorCode (**refineFuncs)(PetscInt, DM, Vec, PetscInt *, void *);
} AppCtx;

static PetscErrorCode RefineTopEdge(PetscInt c, DM cdm, Vec coordinates, PetscInt *label, void *ictx)
{
  AppCtx         *ctx = (AppCtx*)ictx;
  PetscInt       csize;
  PetscScalar    *coords = NULL;
  PetscReal      y3, ytop = ctx->ytop;
  PetscErrorCode ierr;

  /* Get Coordinates */
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  y3 = PetscRealPart(coords[5]);

  if (y3 == ytop) {*label = DM_ADAPT_REFINE;}
  else {*label = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode RefineRightEdge(PetscInt c, DM cdm, Vec coordinates, PetscInt *label, void *ictx)
{
  AppCtx         *ctx = (AppCtx*)ictx;
  PetscInt       csize;
  PetscScalar    *coords = NULL;
  PetscReal      x2, xright = ctx->xright;
  PetscErrorCode ierr;

  /* Get Coordinates */
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  x2 = PetscRealPart(coords[2]);

  if (x2 == xright) {*label = DM_ADAPT_REFINE;}
  else {*label = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode RefineCircumference(PetscInt c, DM cdm, Vec coordinates, PetscInt *label, void *ictx)
{
  AppCtx         *ctx = (AppCtx*)ictx;
  PetscInt       csize, xcenter = ctx->xcenter, ycenter = ctx->ycenter, p = ctx->p;
  PetscScalar    *coords = NULL;
  PetscReal      min, x1, x2, x3, x4, y1, y2, y3, y4, diag, S1, S2, S3, S4;
  PetscErrorCode ierr;

  /* Get Coordinates */
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  x1 = PetscRealPart(coords[0]); x2 = PetscRealPart(coords[2]);
  x3 = PetscRealPart(coords[4]); x4 = PetscRealPart(coords[6]);
  y1 = PetscRealPart(coords[1]); y2 = PetscRealPart(coords[3]);
  y3 = PetscRealPart(coords[5]); y4 = PetscRealPart(coords[7]);

  diag = PetscSqrtReal(PetscSqr(x3-x1) + PetscSqr(y3-y1));
  /* SDF at each cell vertex */
  S1 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x1) + PetscSqr(ycenter-y1)) - p);
  S2 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x2) + PetscSqr(ycenter-y2)) - p);
  S3 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x3) + PetscSqr(ycenter-y3)) - p);
  S4 = PetscAbsReal(PetscSqrtReal(PetscSqr(xcenter-x4) + PetscSqr(ycenter-y4)) - p);

  min = PetscMin(PetscMin(S1, S2), PetscMin(S3, S4));

  if (min<diag) {*label = DM_ADAPT_REFINE;}
  else {*label = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode SetAdaptRefineLabel(DM *preforest, DMLabel *adaptLabel, PetscErrorCode (*GetLabelValue)(PetscInt, DM, Vec, PetscInt *, void *), void *ctx)
{
  DM             cdm, postforest;
  Vec            coordinates;
  PetscInt       c, cstart, cend, label;
  PetscErrorCode ierr;

  ierr = DMForestGetCellChart(*preforest, &cstart, &cend);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*preforest, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(*preforest, &coordinates);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",adaptLabel);CHKERRQ(ierr);

  for (c = cstart; c < cend; ++c)
  {
    ierr = GetLabelValue(c, cdm, coordinates, &label, ctx);CHKERRQ(ierr);
    ierr = DMLabelSetValue(*adaptLabel, c, label);CHKERRQ(ierr);
  }

  /* Apply adaptLabel to the forest and set up */
  ierr = DMForestTemplate(*preforest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
  ierr = DMDestroy(preforest);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(postforest,*adaptLabel);CHKERRQ(ierr);
  ierr = DMSetUp(postforest);CHKERRQ(ierr);
  ierr = DMForestTemplate(postforest, PETSC_COMM_WORLD, preforest);CHKERRQ(ierr);
  ierr = DMDestroy(&postforest);CHKERRQ(ierr);
  ierr = DMSetUp(*preforest);CHKERRQ(ierr);

  return(0);
}

static PetscErrorCode SetupRefineFuncs(DM dm, AppCtx *user)
{
  user->refineFuncs[0] = RefineCircumference;
  user->refineFuncs[1] = RefineTopEdge;
  user->refineFuncs[2] = RefineRightEdge;

  return(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmDist, base, forest;
  PetscInt	 dim = 2, n, nrefine = 4;
  PetscReal      lower[2] = {0,0}, upper[2] = {16,8};
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel        adaptLabel = NULL;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Dimension of mesh */
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Refinement Level */
  ierr = PetscOptionsGetInt(NULL,NULL, "-nrefine", &nrefine, NULL);CHKERRQ(ierr);
  /* Radius of circle */
  user.p = 1;
  ierr = PetscOptionsGetInt(NULL,NULL, "-p", &user.p, NULL);CHKERRQ(ierr);
  /* Calculate center coordinates */
  /* ctx.xcenter = (upper[0]+lower[0])/2; ctx.ycenter = (upper[1]+lower[1])/2; */
  user.xcenter = 2; user.ycenter = 2;
  user.ytop = upper[1]; user.xright = upper[0];

  /* Create a base DMPlex mesh */
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, NULL, lower, upper, NULL, interpolate, &base);CHKERRQ(ierr);
  ierr = DMPlexDistribute(base, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist)
  {
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    base = dmDist;
  }
  ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);
  /* Covert Plex mesh to Forest and destroy base */
  ierr = DMCreate(PETSC_COMM_WORLD, &forest);CHKERRQ(ierr);
  ierr = DMSetType(forest, (dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(forest, base);CHKERRQ(ierr);
  ierr = DMSetUp(forest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(forest, &user);CHKERRQ(ierr);
  ierr = PetscMalloc1(3, &user.refineFuncs);CHKERRQ(ierr);
  ierr = SetupRefineFuncs(forest, &user);CHKERRQ(ierr);

  /* Refinement Loop */
  for (n = 0; n < nrefine; ++n)
  {
    for (PetscInt i = 0; i < 3; ++i)
    {
      /* Call Refinement Functions */
      ierr = SetAdaptRefineLabel(&forest, &adaptLabel, user.refineFuncs[i], &user);CHKERRQ(ierr);
    }
  }

  /* Convert Forest back to Plex for visualization */
  ierr = DMConvert(forest, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&forest);CHKERRQ(ierr);

  /* DM View */
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Petsc Viewer */
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

  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

