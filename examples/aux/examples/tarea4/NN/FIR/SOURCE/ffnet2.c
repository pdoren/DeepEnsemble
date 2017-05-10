/*-----------------------------------------------------------------------------


ffnet2 feeds a sequence of input vectors (in InpVects) through a 2 layer 
neural network specified by the user (in W1,B1,W2,B2) to produce a sequence of
output vectors (OUTPUT). The hidden layer may have 'taps' on its inputs; i.e., 
it can take numTaps past values of that input. The number of taps is determined
automatically from both the number of input vectors and the number of 
weights in the hidden layer (W1).

Eric Wan, Alex Nelson, Thyagarajan Srinivasan  (9/95)
ericwan@eeap.ogi.edu



Structure of Matlab call :

  [OUTPUT] = ffnet2(W1,B1,W2,B2,InpVects);

W?: mxn  (Weight Matrix)            m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?: mx1  (Bias Matrix)              m: nodes in the layer
                          B[i,1] is the bias weight for the ith node in the 
			  layer.

InpVects: numIn x K (Input Vectors) K: number of input vectors in sequence
                                    numIn: number of network inputs
                          InpVects[i,j] is the ith input from the jth input 
			  vector. Column vector j contains the values of all 
			  input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
			  in the hidden layer, then each hidden node will take 
			  values from t columns 'simultaneously'.

OUTPUT: numOut x N (Output Vectors) N: number of output patterns 
                                   numOut: number of network outputs
			  OUTPUT[i,j] is the output of the ith output node at 
			  time k=j. Hence, the jth column vector contains all 
			  the outputs at time k=j. Notice that there will 
			  generally be fewer output vectors than input vectors.
			  The number of columns N, equals the number of input 
			  vectors minus the number of taps;i.e. N=K-t
-----------------------------------------------------------------------------*/
			     /* INCLUDE FILES */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"


			     /* MACROS */
#define FN1(x) (tanh(x))	/* Squashing Function for Hidden Nodes */
#define FN2(x) (x)		/* Squashing Function for Output Nodes */
				/* The following Macros convert the matrix 
				   indices (i,j) into a form useable by the 
				   MATLAB matrix format. This is required 
				   because the matrices used, e.g. W1 and W2, 
				   are infact pointers to a MATLAB object. */
#define c2MAT(i,j,ROWS) (i + j*ROWS)
#define c2MAT_tap(i,j,tap,ROWS,rowsXcolms) (i+ j*ROWS + tap*rowsXcolms)

			     /* GLOBAL VARIABLES */
int numIn;			/* # of input nodes in network */
int numOut;			/* # of output nodes in network */
int numHid;			/* # nodes in in1etermediate layer */
double *W1 =NULL;		/* First layer Weights   */
double *B1 =NULL;		/* First layer Bias      */
double *W2 =NULL;		/* Second Layer Wieghts  */
double *B2 =NULL;		/* Second Layer Bias     */
double *InpVects =NULL;		/* Input Vectors  */
double *netOut;			/* network output */
double *hidOut =NULL;		/* Output at intermediate layer */
int numInpVect;			/* the number of input vectors */
int numTaps;			/* the # of taps */

        		     /* FUNCTION PROTOTYPES */
void forwardpass(void);
/*
void InitGlobals(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);
*/


/* Gateway Routine------------------------------------------------------------ 
     This function creates the interface with the world of MATLAB. The 
 arguments are the sizes of, and pointers to, arrays of the left and right hand
 side matrices entered by the MATLAB user. InitGlobals is called to check the 
 arguments for consistancy, and then to use them to determine the values of the
 Global Variables. It also assigns global pointers to the MATLAB objects and 
 allocates space for the output matrix, and hidden node outputs. Finally, 
 forwardpass is called to generate the output values. */
void mexFunction(
		int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
  
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
  int colms,rows;		/* num of colms and rows of various matrices */

 /* Check and Load global architecture parameters */
  if(nrhs!=5)
  mexErrMsgTxt("5 Arguments needed on RHS: W1, B1, W2, B2, InpVects.");

  if(nlhs!=1)
  mexErrMsgTxt("1 Argument needed on LHS: Output matrix");

  numIn = mxGetM(prhs[4]);	/* num of input nodes = num rows in InpVects */
  numHid = mxGetM(prhs[0]);	/* number of hidden nodes = num rows of W1 */
  colms = mxGetN(prhs[0]);	/* number of colms of W1 */

  if((colms % numIn) != 0)	/* num of colms of W1 shld be mult of numIn */
    mexErrMsgTxt("Invalid W1!\n");
  numTaps = colms /numIn;	/* ... and the num of Taps is that multiple */

  colms = mxGetN(prhs[2]);	/* number of colms of W2... */
  if(colms != numHid)		/* ...should equal the num of Hidden Nodes */
  mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");

  numOut = mxGetM(prhs[2]);	/* the number of outputs = num rows of W2 */

  rows = mxGetM(prhs[1]);       /* the number of rows of B1 */
  colms =mxGetN(prhs[1]);	/* the number of colms of B1 */
  if(rows!=numHid)
     mexErrMsgTxt("B1 must have the same number of rows as W1\n");
  else if(colms!=1)
     mexErrMsgTxt("B1 must have only one column.\n");

  rows = mxGetM(prhs[3]);	/* the number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* the number of colms of B2 */
  if(rows!=numOut)
     mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  else if(colms!=1)
     mexErrMsgTxt("B2 must have only one column.\n");

  numInpVect=mxGetN(prhs[4]);	/* the number of input vectors */
  if(numInpVect<numTaps)
    mexErrMsgTxt("Error! The number of input vectors must be greater than the number of taps!\n");

 /* Assigning pointers to internal variables */
  W1 = mxGetPr(prhs[0]);
  B1 = mxGetPr(prhs[1]);
  W2 = mxGetPr(prhs[2]);
  B2 = mxGetPr(prhs[3]);
  InpVects=mxGetPr(prhs[4]);

 /* Allocate Memory for Hidden node outputs */
  hidOut=(double *) mxCalloc(numHid , sizeof(double));
  if(hidOut == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

 /* Create matrix for the return Argument */
  plhs[0] = mxCreateDoubleMatrix(numOut,numInpVect,mxREAL); /* OUTPUT */
  netOut=mxGetPr(plhs[0]);

/*
}
*/

  forwardpass();
}

/* Forward Pass --------------------------------------------------------------
      Computes the output vector for every input pattern in the matrix of input
 vectors. Note that t input vectors are used simultaneously, where t is the 
 number of taps of each input node. For this reason, the iteration starts on 
 vector t-1, so the delayed inputs will use vectors t-2,t-3,...,0.  */
void forwardpass()
{ int i,j,k,t;
  int rows,clmns;
  double fin_out;
  int numHidXnumIn;
  int tappedi;
    
/*  printf("Inputs: %d\nHidden Nodes: %d\nOutputs: %d\n",numIn,numHid,numOut);
  printf("Input Vectors: %d\nTaps: %d\n",numInpVect,numTaps);*/

  /* Begin forward pass */
  numHidXnumIn = numHid*numIn;
  for(i=numTaps-1;i<numInpVect;i++)/*FOR all input vectors */
  {
    for(k=0;k<numHid;k++)	/*    FOR each node in the hidden layer */
    {
      hidOut[k]=B1[k];		/*      the bias */
      for(t=0;t<numTaps;t++)	/*      FOR each tap of the node */
      { tappedi=i-t;
        for(j=0;j<numIn;j++)	/*         weighted sum of inputs */
          hidOut[k] += InpVects[c2MAT(j,tappedi,numIn)] * 
	               W1[ c2MAT_tap(k,j,t,numHid,numHidXnumIn) ];
      }
      hidOut[k]=FN1(hidOut[k]); /*      sigmoid of weighted sum */
    }
  
    for(k=0;k<numOut;k++)	/*    FOR each node in the output layer */
    {
      fin_out=B2[k];		/*      start with the bias term */
      for(j=0;j<numHid;j++)	/*      FOR each hidden node  */
        fin_out += hidOut[j] * W2[c2MAT(k,j,numOut)]; /* compute inner prod. */
      netOut[c2MAT(k,i,numOut)] = FN2(fin_out); /* kth output for ith pattern*/
    }
  }

}


