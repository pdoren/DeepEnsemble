/*----------------------------------------------------------------------------
This code returns the Jacobian of a two layer neural network
(currently for only one output)


Eric Wan
Alex T. Nelson (5/96)
atnelson@eeap.ogi.edu

Structure of Matlab call :

   J = jacob(W1R,B1R,W2R,B2R,X);

INPUTS:
-------

W?R: mxn  (Initial Weight Matrix)   m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?R: mx1  (Initial Bias Matrix)     m: nodes in the layer
                          B[i,1] is the bias weight for the ith node in the 
			  layer.

X  : tx1  (Input Vector of Delays)  t: number of taps (delays)
                          X holds the most recent t values of the input time
			  series. X(k) is the value of the input vector at 
			  tap k.
			  This code is written for a single input line with
			  t tap-delays.
OUTPUTS: 
--------
J  : txt  (Jacobian Matrix)         t: number of taps will also equal the
                                       number of outputs.
			  This is the Matrix of partial derivatives of the
			  output with respect to the input. J(i,j) is the
			  partial of the ith output w.r.t. the jth input.

-----------------------------------------------------------------------------*/
				/* INCLUDE FILES  */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"

				/* MACROS */
#define FN1(x) (tanh(x))	/* hidden layer transfer function */
#define FN2(x) (x)		/* output layer transfer function */
#define dFN1(x) (1.0 -(x*x))	/* derivative of FN1 wrt x */
#define dFN2(x) (1)		/* derivative of FN2 wrt x */
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
int numTaps;			/* the # of taps */
double *W1R =NULL;		/* First layer Weights - pre_training */
double *B1R =NULL;		/* First layer Bias - pre_training */
double *W2R =NULL;		/* Second Layer Wieghts - pre_training */
double *B2R =NULL;		/* Second Layer Bias - Pre_training */
double *X =NULL;		/* Input Vector */
int *err=NULL;			/* Output Error */
double *netOut=NULL;		/* pointer to network outputs */
double *hidOut =NULL;		/* Output at intermediate layer */
double *deltaOut=NULL;		/* Network output error */
double *deltaHid=NULL;		/* delta at the first layer */
double *J = NULL;		/* pointer to the Jacobian Matrix */

			     /* FUNCTION PROTOTYPES */
/*
void InitGlobals(int nlhs,Matrix *plhs[],int nrhs,Matrix *prhs[]);
*/
void compute_Jacobian(void);
void fwd_pass(void);
void back_prop(void);

/*-------------------------------------------------------------------------*/
/* Gateway Routine */
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

 /* Error Checking */
 /* Error Checking */

  if(nrhs!=5)
    mexErrMsgTxt("5 Args needed on RHS: W1,B1,W2,B2,X");
  if(nlhs!=1)
    mexErrMsgTxt("1 Argument needed on LHS: J");
  
  numIn = 1;			/* num of input nodes = 1 */
  numHid = mxGetM(prhs[0]);	/* number of hidden nodes = num rows of W1 */
  colms = mxGetN(prhs[0]);	/* number of colms of W1 */

  if((colms % numIn) != 0)	/* num of colms of W1 shld be mult of numIn */
    mexErrMsgTxt("Invalid W1!\n");
  numTaps = colms /numIn;	/* ... and the num of Taps is that multiple */

  colms = mxGetN(prhs[2]);	/* number of colms of W2... */
  if(colms != numHid)		/* ...should equal the num of Hidden Nodes */
    mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");
  
  numOut = mxGetM(prhs[2]);	/* the number of outputs = num rows of W2 */
  if(numOut != 1)
    mexErrMsgTxt("Currently only works for 1 output");

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
  
  rows = mxGetM(prhs[4]);	/* the length of the input vector */
  if(rows != numTaps)
    mexErrMsgTxt("Input vector length must equal the number of taps\n");
  if(mxGetN(prhs[4]) != 1)
    mexErrMsgTxt("Only one input (column) vector allowed.\n");

 /* Allocate Memory and Assign Pointers */
  allocReturnMatrices(plhs);	   /* Create Space for the return Arguments */
  assignMatrixPointers(plhs, prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */
/*
}
*/
   compute_Jacobian();
}

/*-------------------------------------------------------------------------*/
/* COMPUTE JACOBIAN */
void compute_Jacobian(void)
{
  fwd_pass();			/* do forward pass through net */
  back_prop();			/* compute deltas */
}

/* Forward Pass ---------------------------------------------------------------
  Takes the most recent numTaps input vectors and propagates the signal 
 forward through the network to produce an output vector. */
void fwd_pass()
{
 int j,k,t;
 int tmp;
 double fin_out;
 int numHidXnumIn = numHid*numIn;

 /* output of first layer */
  for(k=0;k<numHid;k++)		/* FOR each hidden node */
  {
    fin_out=B1R[k];		/*  the bias */
    for(t=0;t<numTaps;t++)	/*  FOR all taps on all inputs */
    {
      for(j=0;j<numIn;j++){	/*  take weighted sum of inputs */
	fin_out+= X[c2MAT(j,t,numIn)] * 
	          W1R[c2MAT_tap(k,j,t,numHid,numHidXnumIn)];
/*        printf("%f \n",X[c2MAT(j,t,numIn)] );  */
      }
    }
    hidOut[k]=FN1(fin_out);	/* output is sigmoid of weighted sum */
  }
 
 /* output of second layer */
  for(k=0;k<numOut;k++)		/* FOR each output node */
  {
    fin_out=B2R[k];
    for(j=0;j<numHid;j++)	/* compute weighted sum of hid outputs */
      fin_out += hidOut[j] * W2R[c2MAT(k,j,numOut)];
    netOut[k] = FN2(fin_out);	/* output is linear function */
  }
}

/* Backprop -------------------------------------------------------------------
   Computes the errors in the output layer (and adds them to the cumulative
  squared errors), then backpropagates these errors to form the delta values
  in the hidden nodes */
void back_prop()
{
  int i,j,t,k;
  double delta;
  int numHidXnumIn = numHid*numIn;


    k = 0;
    deltaOut[0]= 1.0;

    for(i=0;i<numHid;i++)
    {
      delta=0.0;
      /* calculating the back propagated delta */
      delta += deltaOut[0] * W2R[c2MAT(0,i,numOut)];
      deltaHid[i]= delta * dFN1(hidOut[i]);
    }


      for(t=0;t<numTaps;t++)
      {
	delta=0.0;
	/* calculating the back propagated delta */
	for(j=0;j<numHid;j++)
	  delta += deltaHid[j] * W1R[c2MAT_tap(j,0,t,numHid,numHidXnumIn)];

	J[c2MAT(k,t,numOut)]= delta; 

      }
  
}


/*----------------------------------------------------------------------*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numOut,(numTaps*numIn), mxREAL); /* creating J */
}

/*----------------------------------------------------------------------*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1R = mxGetPr(prhs[0]);
  B1R = mxGetPr(prhs[1]);
  W2R = mxGetPr(prhs[2]);
  B2R = mxGetPr(prhs[3]);
  X = mxGetPr(prhs[4]);

  J = mxGetPr(plhs[0]);
}

/*----------------------------------------------------------------------*/
allocInternalMemory(void)
{
  hidOut=(double *) mxCalloc(numHid , sizeof(double));
  if(hidOut == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  netOut=(double *) mxCalloc(numOut , sizeof(double));
  if(netOut ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  err=(int *) mxCalloc(numOut , sizeof(int));
  if(err ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  deltaHid = (double *) mxCalloc (numHid , sizeof(double));
  if(deltaHid==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  deltaOut = (double *) mxCalloc (numOut , sizeof(double));
  if(deltaOut==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

}

















