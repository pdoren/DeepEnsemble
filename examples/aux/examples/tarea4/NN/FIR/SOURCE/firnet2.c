/*-FIRNET2.C --------------------------------------------------------------

 firnet2 feeds a sequence of input vectors (in InpVects) through a 2 layer 
neural network specified by the user (W1,B1,W2,B2) to produce a sequence 
of output vectors (OUTPUT). The hidden layer nodes may have 'taps' on their 
inputs; i.e., they can take numTaps past values of that input. The number of
taps is determined automatically from the both the number of input vectors 
and the dimensions of the weight matrices supplied.

Eric Wan, Alex Nelson, Thyagarajan Srinivasan  (9/95)
ericwan@eeap.ogi.edu


Structure of Matlab call :

  [OUTPUT] = firnet2(W1R,B1R,W2R,B2R,InpVects);

INPUTS:
-------

W?: mxn  (Initial Weight Matrix)    m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?: mx1  (Initial Bias Matrix)      m: nodes in the layer
                          B[i,1] is the bias weight for the ith node in the 
			  layer. 

InpVects: numIn x K (Input Vectors) K: number of input vectors in sequence
                                      numIn: number of network input nodes.
                          InpVects[i,j] is the ith input from the jth input 
			  vector. Column vector j contains the values of all 
			  of the input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
			  in the hidden layer, then each hidden node will take 
			  values from t columns 'simultaneously'.

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
#define FN1(x) (tanh(x))	/* hidden layer transfer function */
#define FN2(x) (x)		/* output layer transfer function */
				/* The following Macros convert the matrix 
				   indices (i,j) into a form useable by the 
				   MATLAB matrix format. This is required 
				   because the matrices used, e.g. W1 and W2, 
				   are infact pointers to a MATLAB object. */
#define c2MAT(i,j,ROWS) (i + j*ROWS)
#define c2MAT_tap(i,j,taps,ROWS,rowsXcolms) (i+ j*ROWS + taps*rowsXcolms)

			     /* GLOBAL VARIABLES */
int numIn;			/* # of input nodes in network */
int numOut;			/* # of output nodes in network */
int numHid;			/* # nodes in inetermediate layer */
double *W1 =NULL;		/* First layer Weights  */
double *B1 =NULL;		/* First layer Bias  */
double *W2 =NULL;		/* Second Layer Wieghts */
double *B2 =NULL;		/* Second Layer Bias  */
double *InpVects =NULL;		/* Input Vectors  */
double *netOut=NULL;		/* pointer to network outputs */
double **hidOut =NULL;		/* Output at intermediate layer [node][tap]*/
int numInpVects;		/* the number of input vectors */
int numTaps1;			/* the number of taps in 1st layer*/
int numTaps2;			/* the number of taps in 2nd layer*/
int mask;

			     /* FUNCTION PROTOTYPES */

/*
void InitGlobals(int nlhs,mxArray *plhs[],int nrhs, const mxArray *prhs[]);
*/
void forwardPass(void);
void fwdPassLayer1(int trg_index);
void fwdPassLayer2(int trg_index);

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
void InitGlobals(int nlhs, mxArray *plhs[],
		int nrhs, mxArray *prhs[])
{
*/
  int i,colms,rows;		/* num of colms and rows of various matrices */
  int numWeights;

 /* CHECK FOR ERRORS IN ARGUMENTS */
  if(nrhs!=5)
    mexErrMsgTxt("5 Arguments needed on RHS:  W1, B1, W2, B2, InpVects.\n");
  if(nlhs!=1)
    mexErrMsgTxt("1 Argument needed on LHS: OUTPUT.\n");
  
  numIn = mxGetM(prhs[4]);	/* num of input nodes = num rows in InpVects */
  numHid = mxGetM(prhs[0]);	/* number of hidden nodes = num rows of W1 */
  colms = mxGetN(prhs[0]);	/* number of colms of W1 */

  if((colms % numIn) != 0)	/* num of colms of W1 shld be mult of numIn */
    mexErrMsgTxt("Number of columns of W1 & rows of InpVects incompatible\n");
  numTaps1 = colms /numIn;	/* num of taps betwn In & Hid is that multipl*/

  colms = mxGetN(prhs[2]);	/* number of colms of W2... */
   if((colms % numHid) !=0)	/* num of colms of W2 shld be mult of numHid*/
    mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");
  numTaps2 = colms / numHid;	/* num of taps betwn Hid & Out is the multplr*/

  numOut = mxGetM(prhs[2]);	/* the number of outputs = num rows of W2 */
  
  rows = mxGetM(prhs[1]);       /* the number of rows of B1 */
  colms =mxGetN(prhs[1]);	/* the number of colms of B1 */
  if(rows!=numHid)
    mexErrMsgTxt("B1 must have the same number of rows as W1\n");
  if(colms!=1)
    mexErrMsgTxt("B1 must have only one column.\n");
  
  rows = mxGetM(prhs[3]);	/* the number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* the number of colms of B2 */
  if(rows!=numOut)
    mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  if(colms!=1)
    mexErrMsgTxt("B2 must have only one column.\n");
  
  numInpVects=mxGetN(prhs[4]);	/* the number of input vectors */
  if(numInpVects <= (numTaps1 + numTaps2))
  {  mexErrMsgTxt("Error! The number of input vectors must be greater \
than the total number of taps!\n");
  }
  rows = mxGetM(prhs[4]);	/* the length of the input vectors */
  if(rows != numIn)
    mexErrMsgTxt("Input vector length must equal number of input nodes.\n");

 /* FORM CIRCULAR INDEX */
  if(numTaps2>1)
  {
    mask = 1 << ((int) ( log((double)(numTaps2 - 1)) / 0.69314718 ) + 1);
    mask -= 1;
  }
  else
    mask = 0;

 /* ALLOCATE MEMORY AND ASSIGN POINTERS */
  allocReturnMatrices(plhs);	   /* Create Space for the return Arguments */
  assignMatrixPointers(plhs,prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */

/*  printf("\nInputs: %d\nHidden Nodes: %d\nOutput Nodes: %d\n",
	 numIn,numHid,numOut);
  printf("Training Vectors: %d\n",numInpVects);
  printf("Second Layer Taps: %d  First Layer Taps: %d \n",
	 numTaps2, numTaps1);
*/
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

  for(i= (numTaps1 - 1); i< (numTaps1+numTaps2 -2); i++)
    fwdPassLayer1(i);		/* prime the second layer of taps */

  for(;i<numInpVects;i++)	/* go over the remaining input vectors */
  {
    fwdPassLayer1(i);
    fwdPassLayer2(i);
  }
}

/* Forward Pass Layer 1 -------------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the first set of taps and weights for one time step to 
 produce an output vector for the hidden layer. */
void fwdPassLayer1(int trg_index)
{
  int j,k,t;
  int tmp,tmp1;
  double fin_out;
  int numHidXnumIn = numHid*numIn;

  tmp1 = trg_index & mask;

  for(k=0; k<numHid; k++)	/* FOR each node in the hidden layer */
  {
    fin_out=B1[k];		
    for(t=0; t<numTaps1; t++)	/*  FOR all taps on all inputs */
    {
      tmp = trg_index - t;
      for(j=0; j<numIn; j++)	/* weighted sum of inputs */
        fin_out+= InpVects[c2MAT(j,tmp,numIn)] *
	          W1[c2MAT_tap(k,j,t,numHid,numHidXnumIn)];
    }
    hidOut[k][tmp1]=FN1(fin_out); /*sigmoid of weighted sum */
  }
}

/* Forward Pass Layer 2 -------------------------------------------------------
  Takes the (numTaps2) most recent output vectors of the hidden layer, and 
 propagates the signal forward through the 2nd set of taps and weights to 
 produce an output vector for the network. */
void fwdPassLayer2(int trg_index)
{
  int j,k,t;
  int tmp;
  double fin_out;
  int numOutXnumHid = numOut*numHid;
  
  for(k=0; k<numOut; k++)	/* FOR each output */
  {
    fin_out = B2[k];
    for(t=0; t<numTaps2; t++)	/*   FOR all taps on each hidden node output */
    {
      tmp = (trg_index-t) & mask;
      for(j=0; j<numHid; j++)	/*      Compute the inner product */
        fin_out += hidOut[j][tmp] * W2[c2MAT_tap(k,j,t,numOut,numOutXnumHid)];
    }
    netOut[c2MAT(k,trg_index,numOut)] = FN2(fin_out);
  }
}



/* Allocate Return Matrices -------------------------------------------------
  Creates space for the return matrices (MATLAB matrices).*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numOut,numInpVects, mxREAL);	  /* creating OUTPUT */
}

/* Assign Matrix Pointer -----------------------------------------------------
  Assigns C Pointers to the Matlab matrix arguments.*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1 = mxGetPr(prhs[0]);
  B1 = mxGetPr(prhs[1]);
  W2 = mxGetPr(prhs[2]);
  B2 = mxGetPr(prhs[3]);
  InpVects = mxGetPr(prhs[4]);

  netOut = mxGetPr(plhs[0]);
}

/* Allocate Internal Memory ---------------------------------------------------
  Creates space for data internal to the program */
 allocInternalMemory(void)
{ int i;
  
  hidOut=(double **) mxCalloc(numHid , sizeof(double *));
  if(hidOut == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  for(i=0;i<numHid;i++)
  {
    hidOut[i] = (double *) mxCalloc ((mask + 1) , sizeof(double));
    if(hidOut[i]==NULL)
      mexErrMsgTxt("Unable to Allocate memory!\n");
  }
}
