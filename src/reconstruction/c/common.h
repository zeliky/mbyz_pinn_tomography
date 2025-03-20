#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>

/* Constants */
#define eps 2.2204460492503131e-16
#define doublemax 1e50
#define INF 2e50

#define listINF 2.345e50

/* Macros */
#ifndef min
#define min(a,b)        ((a) < (b) ? (a): (b))
#endif
#ifndef max
#define max(a,b)        ((a) > (b) ? (a): (b))
#endif

/* Function Prototypes */
int minarray(double *A, int l);
int maxarray(double *A, int l);
double pow2(double val);
int iszero(double a);
int isnotzero(double a);
void roots(double* Coeff, double* ans);
bool IsFinite(double x);
bool IsInf(double x);
bool IsListInf(double x);

int mindex2(int x, int y, int sizx);
int mindex3(int x, int y, int z, int sizx, int sizy);

bool isntfrozen2d(int i, int j, int *dims, bool *Frozen);
bool isfrozen2d(int i, int j, int *dims, bool *Frozen);
bool isntfrozen3d(int i, int j, int k, int *dims, bool *Frozen);
bool isfrozen3d(int i, int j, int k, int *dims, bool *Frozen);

void initialize_list(double **listval, int *listprop);
void destroy_list(double **listval, int *listprop);
void list_add(double **listval, int *listprop, double val);
int list_minimum(double **listval, int *listprop);
void listupdate(double **listval, int *listprop, int index, double val);
void remove_min(double **listval, int *listprop, int *i, int *j);
void list_remove_replace(double **listval, int *listprop, int index);
extern void list_remove(double **listval, int *listprop, int index);
double CalculateDistance(double *T, double Fij, int *dims, int i, int j, bool usesecond, bool usecross, bool *Frozen);

extern int dx[4];
extern int dy[4];

#endif /* COMMON_H */

/*#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Define INF only if not already defined
#ifndef INF
//#define INF 1e20  // Reduced from 1e50 to prevent overflow issues
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

// Function declarations
extern double pow2(double val);
extern void roots(double* Coeff, double* ans);
extern int minarray(double *A, int l);
extern int maxarray(double *A, int l);
extern int mindex2(int x, int y, int sizx);
extern bool isfrozen2d(int i, int j, int *dims, bool *Frozen);
extern bool isntfrozen2d(int i, int j, int *dims, bool *Frozen);
extern bool IsInf(double x);
extern bool IsFinite(double x);
extern bool IsListInf(double x);

// List handling functions
extern void initialize_list(double **listval, int *listprop);
extern void destroy_list(double **listval, int *listprop);
extern void list_add(double **listval, int *listprop, double val);
extern int list_minimum(double **listval, int *listprop);
extern void listupdate(double **listval, int *listprop, int index, double val);
extern void list_remove_replace(double **listval, int *listprop, int index);


// Travel-time calculation function
extern double CalculateDistance(double *T, double Fij, int *dims, int i, int j, bool usesecond, bool usecross, bool *Frozen);

#endif  // COMMON_H
 */