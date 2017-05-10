/*-----------------------------------------------------------------------------
 firnet3 feeds a sequence of input vectors (in InpVects) through a 3 layer 
neural network specified by the user (W1,B1,W2,B2,W3,B3) to produce a sequence 
of output vectors (OUTPUT). The hidden layers may have 'taps' on their inputs; 
i.e., they can take numTaps past values of that input. The number of taps is
determined automatically from the both the number of input vectors and the 
dimensions of the weight matrices supplied.


Eric Wan, Alex Nelson, Thyagarajan Srinivasan  (9/95)
ericwan@eeap.ogi.edu


Structure of Matlab call :

  [OUTPUT] = firnet3(W1,B1,W2,B2,W3,B3,InpVects);

INPUTS:
-------

W?: mxn  (Initial Weight Matrix)   m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?: mx1  (Initial Bias Matrix)     m: nodes in the layer
                          B[i] is the bias weight for the ith node in the 
			  layer.  

InpVects: numIn x K (Input Vectors) K: number of input vectors in sequence
                                      numIn: number of network inputs
                          InpVects[i,j] is the ith input from the jth input 
                          vector. Column vector j contains the values of all 
			  of the input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
                          in the first hidden layer, then each node in that 
		          layer will take values from t columns at the same
       			  time.


OUTPUTS: numOut x K (Output Vectors)K: number of input vectors
                       		    N: number of nonzero output vectors
                                    numOut: number of network outputs
			  OUTPUT[i,j] is the ith  output at 
			  time k=j. Hence, the jth column vector contains all
			  the outputs at time k=j. Each row of the 
			  matrix is comprised of an individual time series of 
			  output values, from k=0 to k=K-1. The number of 
			  columns equals the number of input vectors, but
			  Notice that there will generally be fewer nonzero 
			  output vectors than input vectors. The number of 
			  non-zero columns N, equals the number of input 
			  vectors minus the total number of taps plus three;
			  i.e.,  N = K - (t1-1) - (t2-1) - (t3-1).


	Author : Thyagarajan Srinivasan,
		 Dept. of Electrical Engineering & Applied Physics,
		 Oregon Graduate Institute.
		 thyagu@eeap.ogi.edu
		 Sep 23 1995
-----------------------------------------------------------------------------*/
				/* INCLUDE FILES  */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
     
				/* MACROS */
#define FN1(x) (tanh(x))	/* first hidden layer transfer function */
#define FN2(x) (tanh(x))	/* second hidden layer transfer function */
#define FN3(x) (x)		/* output layer transfer function  */

				/* The following Macros convert the matrix 
				   indices (i,j) into a form useable by the 
				   MATLAB matrix format. This is required 
				   because the matrices used, e.g. W1 and W2, 
				   are infact pointers to a MATLAB object. */
#define c2MAT(i,j,ROWS) (i + j*ROWS)
#define c2MAT_tap(i,j,taps,ROWS,rowsXcolms) (i+ j*ROWS + taps*rowsXcolms)

int numIn;			/* num of input nodes in network */
int numOut;			/* num of output nodes in network */
int numHid1;			/* num of nodes in 1st intermediate layer */
int numHid2;			/* num of nodes in 2nd intermediate layer */
double *W1 =NULL;		/* First layer Weights  */
double *B1 =NULL;		/* First layer Bias  */
double *W2 =NULL;		/* Second Layer Weights  */
double *B2 =NULL;		/* Second Layer Bias  */
double *W3 =NULL;		/* Third Layer Weights */
double *B3 =NULL;		/* Third Layer Bias  */
double *InpVects =NULL;		/* Input Vectors */
double *netOut=NULL;		/* network output */
double **hid1Out =NULL;		/* Output at intermediate layer [node][tap]*/
double **hid2Out =NULL;		/* Output at second layer [node][tap] */
int numInpVects;		/* the number of input vectors */
int numTaps1;			/* the num of taps in 1st layer*/
int numTaps2;			/* the num of taps in 2nd layer*/
int numTaps3;			/* the num of taps in 3rd layer */
int mask2;			/* used for circular indexing of layer 2 */
int mask3;			/* used for circular indexing of layer 3 */

/*
void InitGlobals(int nlhs,Matrix *plhs[],int nrhs,Matrix *prhs[]);
*/
void forwardPass(void);
void fwdPassLayer1(int trg_index);
void fwdPassLayer2(int trg_index);
void fwdPassLayer3(int trg_index);

/* Gateway Routine ----------------------------------------------------------*/

void mexFunction(
		 int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
/*
  InitGlobals(nlhs,plhs,nrhs,prhs);
*/
    /* InitGlobals -------------------------------------------------------------- 
   First loads the values defining the architecture into the global variables, 
  while making sure these values are self-consistant. Prints error messages
  regarding any inconsistancies. 
   Next, assigns global pointers to the MATLAB objects. 
  Finally, allocates memory for the hidden unit outputs and the network 
  outputs, and assigns global pointers to these new objects. */
/*
void InitGlobals(int nlhs, Matrix *plhs[],
		int nrhs, Matrix *prhs[])
{
*/
  int i,colms,rows;		/* num of colms and rows of various matrices */
  int numWeights;

 /* CHECK FOR ERRORS IN ARGUMENTS */
  if(nrhs!=7)
    mexErrMsgTxt("7 Arguments needed on RHS: W1, B1, W2, B2, W3, B3, InpVects.\n");
  if(nlhs!=1)
    mexErrMsgTxt("1 Argument needed on LHS: OUTPUTS.\n");
  
  numIn = mxGetM(prhs[6]);	/* num of input nodes = num rows in InpVects */
  numHid1 = mxGetM(prhs[0]);	/* num of hidden nodes = num rows of W1 */

  colms = mxGetN(prhs[0]);	/* number of colms of W1 */
  if((colms % numIn) != 0)	/* ...should be multiple of numIn */
    mexErrMsgTxt("Number of columns of W1 & rows of InpVects incompatible\n");
  numTaps1 = colms /numIn;	/* numTaps betw Inp & Hid is that multiple */

  colms = mxGetN(prhs[2]);	/* number of colms of W2*/
  if((colms % numHid1) !=0)	/* ...should be multiple of numHid1 */
    mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");
  numTaps2 = colms / numHid1;	/* num of taps betw Hid & Out is that multpl */

  numHid2 = mxGetM(prhs[2]);	/* num of rows of W2 is size of 2nd hid layer*/
  colms = mxGetN(prhs[4]);	/* num of colms of W3 */
  if((colms % numHid2) !=0)	/* ...should be multiple of numHid2*/
    mexErrMsgTxt("Matrices W2 and W3 are incompatible!\n");
  numTaps3 = colms / numHid2;	/* numTaps3 is the multiplier */

  numOut = mxGetM(prhs[4]);	/* number of outputs = num rows of W3 */

  rows = mxGetM(prhs[1]);	/* number of rows of B1 */
  colms =mxGetN(prhs[1]);	/* number of colms of B1 */
  if(rows!=numHid1)
    mexErrMsgTxt("B1 must have the same number of rows as W1\n");
  if(colms!=1)
    mexErrMsgTxt("B1 must have only one column.\n");

  rows = mxGetM(prhs[3]);	/* number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* number of colms of B2 */
  if(rows!=numHid2)
    mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  if(colms!=1)
    mexErrMsgTxt("B2 must have only one column.\n");

  rows = mxGetM(prhs[5]);	/* number of rows of B3 */
  colms =mxGetN(prhs[5]);	/* number of colms of B3 */
  if(rows!=numOut)
    mexErrMsgTxt("B3 must have the same number of rows as W3\n");
  if(colms!=1)
    mexErrMsgTxt("B3 must have only one column.\n");

  numInpVects = mxGetN(prhs[6]); /* the num of input vectors */
  if(numInpVects <= (numTaps1 + numTaps2 + numTaps3))
    mexErrMsgTxt("Error! The number of input vectors must be greater \
than the total number of taps!\n");
  
 /* FORM CIRCULAR INDICES */
  if((numTaps2+numTaps3) > 2)
  {
    mask2= 1 << ((int) (log((double) (numTaps2+numTaps3-2)) / 0.69314718) + 1);
    mask2-=1;
  }
  else
  mask2 = 0;

  if(numTaps3 > 1)
  {
    mask3= 1 << ((int) (log((double) (numTaps3 - 1)) / 0.69314718) + 1);
    mask3-=1;
  }
  else
  mask3 = 0;

 /* ALLOCATE MEMORY AND ASSIGN POINTERS */
  allocReturnMatrices(plhs);	   /* Create Space for the return Arguments */
  assignMatrixPointers(plhs,prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */

/*  printf("Inputs: %d\t1st Hid. Layer: %d\t2nd Hid. Layer: %d\tOutputs: %d\n",
	 numIn,numHid1,numHid2,numOut);
  printf("1st Layer Taps: %d\t2nd Layer Taps: %d\t3rd Layer Taps: %d\n",
	 numTaps1,numTaps2,numTaps3);
  printf("Input Vectors: %d \n",numInpVects); */
/*
}
*/

  forwardPass();
}

/* Forward Pass ------------------------------------------------------------
  Feeds the inputs through the network. Because of the internal delays in
  the network. Some priming must be done before meaningful outputs are
  produced. */
void forwardPass(void)
{ int i;

  for(i=numTaps1-1;i<(numTaps1+numTaps2-2);i++)
    fwdPassLayer1(i);		/* prime second layer of taps */
  for(;i< (numTaps1+numTaps2+numTaps3-3);i++)
  {
    fwdPassLayer1(i);		/* prime third layer of taps */
    fwdPassLayer2(i);
  }
  for(;i<numInpVects;i++)	/* go over the remaining data */
  { 
    fwdPassLayer1(i);
    fwdPassLayer2(i);
    fwdPassLayer3(i);
  }
}


/* Foward Pass Layer 1 ----------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the first set of taps and weights for one time step to 
 produce an output vector for the first hidden layer. */
void fwdPassLayer1(int trg_index)
{
int j,k,t,index,tmp,tmp1;
double fin_out;
int numHid1XnumIn = numHid1*numIn;

  tmp1 = trg_index & mask2;
  for(k=0;k<numHid1;k++)	/* for each node in the layer */
  {
    fin_out=B1[k];		/* the bias */
    for(t=0;t<numTaps1;t++)
    {
      tmp = trg_index-t;
      for(j=0;j<numIn;j++)	/* weighted sum of inputs */
        fin_out+= InpVects[c2MAT(j,tmp,numIn)] *
	          W1[c2MAT_tap(k,j,t,numHid1,numHid1XnumIn)];
    }
    hid1Out[k][tmp1]=FN1(fin_out); /*sigmoid of weighted sum */
  }
}

/* Foward Pass Layer 2 ----------------------------------------------------
  Takes the (numTaps2) most recent input vectors and propagates the signal 
 forward through the second set of taps and weights for one time step to 
 produce an output vector for the second hidden layer. */
void fwdPassLayer2(int trg_index)
{
int j,k,t,tmp,tmp1;
double fin_out;
int numHid2XnumHid1 = numHid2*numHid1;

  tmp1 = trg_index & mask3;

  for(k=0;k<numHid2;k++)
  {
    fin_out=B2[k];
    for(t=0;t<numTaps2;t++)
    {
      tmp = (trg_index-t) & mask2;
      for(j=0;j<numHid1;j++)
        fin_out += hid1Out[j][tmp] * 
	           W2[c2MAT_tap(k,j,t,numHid2,numHid2XnumHid1)];
    }
    hid2Out[k][tmp1] = FN2(fin_out);
  }
}

/* Foward Pass Layer 3 ----------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the third set of taps and weights for one time step to 
 produce an output vector for the third hidden layer. */
void fwdPassLayer3(int trg_index)
{
int j,k,t,tmp;
double fin_out;
int numOutXnumHid2 = numOut*numHid2;

  for(k=0;k<numOut;k++)
  {
    fin_out=B3[k];
    for(t=0;t<numTaps3;t++)
    {
      tmp = (trg_index-t) & mask3;
      for(j=0;j<numHid2;j++)
        fin_out+=hid2Out[j][tmp] * W3[c2MAT_tap(k,j,t,numOut,numOutXnumHid2)];
    }
    netOut[c2MAT(k,trg_index,numOut)] = FN3(fin_out);
  }
}


/* Allocate Return Matrices -------------------------------------------------
  Creates space for the return matrices (MATLAB matrices).*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numOut,numInpVects, mxREAL); /* creating OUTPUT */
}

/* Assign Matrix Pointer -----------------------------------------------------
  Assigns C Pointers to the Matlab matrix arguments.*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1 = mxGetPr(prhs[0]);
  B1 = mxGetPr(prhs[1]);
  W2 = mxGetPr(prhs[2]);
  B2 = mxGetPr(prhs[3]);
  W3 = mxGetPr(prhs[4]);
  B3 = mxGetPr(prhs[5]);
  InpVects = mxGetPr(prhs[6]);

  netOut = mxGetPr(plhs[0]);
}

/* Allocate Internal Memory ---------------------------------------------------
  Creates space for data internal to the program */
allocInternalMemory(void)
{ int i;
 
  hid1Out=(double **) mxCalloc(numHid1 , sizeof(double *));
  if(hid1Out == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  for(i=0;i<numHid1;i++)
  {
    hid1Out[i]= (double *) mxCalloc ((mask2 +1) , sizeof(double));
    if(hid1Out[i]==NULL)
      mexErrMsgTxt("Unable to Allocate memory!\n");
  }

  hid2Out=(double **) mxCalloc(numHid2 , sizeof(double *));
  if(hid2Out == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  for(i=0;i<numHid2;i++)
  {
    hid2Out[i]= (double *) mxCalloc ((mask3+1) , sizeof(double));
    if(hid2Out[i]==NULL)
      mexErrMsgTxt("Unable to Allocate memory!\n");
  }
}
