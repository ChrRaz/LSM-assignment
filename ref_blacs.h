/*
  Currently used functions in the reference scalapack project

  If you start using more ScaLAPACK routines, please extend this
  list of ScaLAPACK/BLACS methods.
 */
void blacs_get_(int *, int *, int *);
void blacs_gridinit_(int *, char *, int *, int *);
void blacs_gridexit_(int *);
void blacs_gridinfo_(int *, int *, int *, int *, int *);
int numroc_(int *, int *, int *, int *, int *);
int indxl2g_(int *, int *, int *, int *, int *);

void descinit_(int *, int *, int *, int *, int *,
	       int *, int *, int *, int *, int *);

void pdgemr2d_(int *, int *,
	       double *, int *, int *, int *,
	       double *, int *, int *, int *,
	       int *);

