static char help[] = "Adaptively refine mesh using mutiple external refinement functions\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscmath.h>

typedef struct {
  PetscInt       p, xcenter, ycenter, ytop, xright;
} Ctx;

struct _refinement_funcs_struct{
  PetscErrorCode (**refineFuncs)(PetscInt, DM, Vec, PetscInt *, void *);
  Ctx            **ctx;
  PetscInt       n; /* Number of functions */
};

typedef struct _refinement_funcs_struct RefinementFunctions;

static PetscErrorCode RefineTopEdge(PetscInt c, DM cdm, Vec coordinates, PetscInt *label_val, void *ictx)
{
  Ctx            *ctx = (Ctx*)ictx;
  PetscInt       csize;
  PetscScalar    *coords = NULL;
  PetscReal      y3, ytop = ctx->ytop;
  PetscErrorCode ierr;

  /* Get Coordinates */
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  y3 = PetscRealPart(coords[5]);

  if (y3 == ytop) {*label_val = DM_ADAPT_REFINE;}
  else {*label_val = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode RefineRightEdge(PetscInt c, DM cdm, Vec coordinates, PetscInt *label_val, void *ictx)
{
  Ctx            *ctx = (Ctx*)ictx;
  PetscInt       csize;
  PetscScalar    *coords = NULL;
  PetscReal      x2, xright = ctx->xright;
  PetscErrorCode ierr;

  /* Get Coordinates */
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  x2 = PetscRealPart(coords[2]);

  if (x2 == xright) {*label_val = DM_ADAPT_REFINE;}
  else {*label_val = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode RefineCircumference(PetscInt c, DM cdm, Vec coordinates, PetscInt *label_val, void *ictx)
{
  Ctx            *ctx = (Ctx*)ictx;
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

  if (min<diag) {*label_val = DM_ADAPT_REFINE;}
  else {*label_val = DM_ADAPT_KEEP;}

  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, c, &csize, &coords);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode SetAdaptRefineLabel(DM *preforest, DMLabel *adaptLabel, RefinementFunctions *RF)
{
  DM             cdm, postforest;
  Vec            coordinates;
  PetscInt       i, c, cstart, cend, label_val, curr_label;
  PetscErrorCode ierr;

  ierr = DMForestGetCellChart(*preforest, &cstart, &cend);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*preforest, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(*preforest, &coordinates);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",adaptLabel);CHKERRQ(ierr);

  for (c = cstart; c < cend; ++c)
  {
    label_val = DM_ADAPT_KEEP; /* Initialize label = 0 */
    for (i = 0; i < RF->n; ++i)
    {
      ierr = RF->refineFuncs[i](c, cdm, coordinates, &curr_label, RF->ctx[i]);CHKERRQ(ierr);
      if (label_val == DM_ADAPT_REFINE || curr_label == DM_ADAPT_REFINE) {label_val = DM_ADAPT_REFINE;}
      else {label_val = DM_ADAPT_KEEP;}
    }
    ierr = DMLabelSetValue(*adaptLabel, c, label_val);CHKERRQ(ierr);
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

static PetscErrorCode CreateRefinementFunctionStructure(RefinementFunctions *RF)
{
  RF->n = 0;
  RF->refineFuncs = NULL;
  RF->ctx = NULL;

  return(0);
}

static PetscErrorCode DestroyRefinementFunctionStructure(RefinementFunctions *RF)
{
  PetscErrorCode ierr;

  RF->n = 0;
  RF->refineFuncs = NULL;
  RF->ctx = NULL;
  //ierr = PetscFree2(RF->refineFuncs, RF->ctx);CHKERRQ(ierr);

  return(0);
}

static PetscErrorCode AddRefinementFunction(RefinementFunctions *RF, PetscErrorCode (*func)(PetscInt, DM, Vec, PetscInt *, void *), void *ictx)
{
  Ctx            *ctx = (Ctx*)ictx;
  PetscErrorCode ierr;

  ierr = PetscRealloc(RF->n+1, &RF->refineFuncs);CHKERRQ(ierr);
  ierr = PetscRealloc(RF->n+1, &RF->ctx);CHKERRQ(ierr);
  RF->refineFuncs[RF->n] = func;
  RF->ctx[RF->n] = ctx;
  ++RF->n;

  return(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmDist, base, forest;
  PetscInt	 dim = 2, n, nrefine = 4;
  PetscReal      lower[2] = {0,0}, upper[2] = {6,10};
  PetscBool      interpolate = PETSC_TRUE;
  DMLabel        adaptLabel = NULL;
  Ctx            ctx[3];
  PetscErrorCode ierr;
  RefinementFunctions RF;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Dimension of mesh */
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Refinement Level */
  ierr = PetscOptionsGetInt(NULL,NULL, "-nrefine", &nrefine, NULL);CHKERRQ(ierr);

  /* Context for RefineCircumference */
  ctx[0].p = 1;
  ierr = PetscOptionsGetInt(NULL,NULL, "-p", &ctx[0].p, NULL);CHKERRQ(ierr);
  ctx[0].xcenter = 2; ctx[0].ycenter = 2;

  /* Context for RefineTopEdge */
  ctx[1].ytop = upper[1];

  /* Context for RefineRightEdge */
  ctx[2].xright = upper[0];

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

  /* Set Refinement Functions Application Context */
  ierr = DMSetApplicationContext(forest, &RF);CHKERRQ(ierr);

  /* Initialize RF structure and add refinement functions */
  ierr = CreateRefinementFunctionStructure(&RF);CHKERRQ(ierr);
  ierr = AddRefinementFunction(&RF, &RefineCircumference, &ctx[0]);CHKERRQ(ierr);
  ierr = AddRefinementFunction(&RF, &RefineTopEdge, &ctx[1]);CHKERRQ(ierr);
  ierr = AddRefinementFunction(&RF, &RefineRightEdge, &ctx[2]);CHKERRQ(ierr);

  /* Refinement Loop */
  for (n = 0; n < nrefine; ++n)
  {
    /* Set refine labels on each cell using info from refinement functions */
    ierr = SetAdaptRefineLabel(&forest, &adaptLabel, &RF);CHKERRQ(ierr);
  }

  /* Convert Forest back to Plex for visualization */
  ierr = DMConvert(forest, DMPLEX, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&forest);CHKERRQ(ierr);
  ierr = DestroyRefinementFunctionStructure(&RF);CHKERRQ(ierr);

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

