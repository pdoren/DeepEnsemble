
/*----------------------------------------------------------------------------

Returns Weight Jacobians (currently only for single output)

Eric Wan, Alex Nelson, 
ericwan@eeap.ogi.edu


Structure of Matlab call :

   [W1L,B1L,W2L,B2L] = jacobW(W1R,B1R,W2R,B2R,X)

INPUTS:
-------

W?R: mxn  (Initial Weight Matrix)   m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?R: mx1  (Initial Bias Matrix)     m: nodes in the layer
                          B[i,1] is the bias weight for the ith node in the 
			  layer.

X: numIn x K (Input Vectors) K: number of input vectors in sequence
                                      numIn: number of network inputs
                          InpVects[i,j] is the ith input from the jth input 
			  vector. Column vector j contains the values of all 
			  of the input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
			  in the hidden layer, then each hidden node will take 
			  values from t columns 'simultaneously'.

OUTPUTS: 
--------
W?L: mxn (Final Weight Matrix) Weight Jacobian
                          Same as W?R, but contains the weights after the last
			  training epoch.
B?L: mx1 (Final Bias Matrix)
                          Same as B?R, but contains the biases after the last
			  training epoch.

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
double *W1R =NULL;		/* First layer Weights - pre_training */
double *B1R =NULL;		/* First layer Bias - pre_training */
double *W2R =NULL;		/* Second Layer Wieghts - pre_training */
double *B2R =NULL;		/* Second Layer Bias - Pre_training */
double *X =NULL;		/* Input Vector */

double *W1L =NULL;		/* First layer Weights - post_training */
double *B1L =NULL;		/* First layer Bias - post_training */
double *W2L =NULL;		/* Second Layer Wieghts - post_training */
double *B2L =NULL;		/* Second Layer Bias - Post_training */
double *InpVects =NULL;		/* Training Vectors - Input */
double *netOut=NULL;		/* pointer to network outputs */
double *hidOut =NULL;		/* Output at intermediate layer */
double *deltaHid=NULL;		/* delta at the first layer */
double *deltaOut=NULL;		/* Network output error */
int numInpVects;		/* the number of training vectors */
int numTaps;			/* the # of taps */

			     /* FUNCTION PROTOTYPES */
/*
void InitGlobals(int nlhs,Matrix *plhs[],int nrhs,Matrix *prhs[]);
*/
void train(void);
void fwd_pass(void);
void weight_update(void);
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

  if(nrhs!=5)
    mexErrMsgTxt("7 Args needed on RHS: W1,B1,W2,B2,InpVects");
  if(nlhs!=4)
    mexErrMsgTxt("5 Arguments needed on LHS: W1, B1, W2, B2 ");
  
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
  if(colms!=1)
    mexErrMsgTxt("B1 must have only one column.\n");
  
  rows = mxGetM(prhs[3]);	/* the number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* the number of colms of B2 */
  if(rows!=numOut)
    mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  if(colms!=1)
    mexErrMsgTxt("B2 must have only one column.\n");
  
  rows = mxGetM(prhs[4]);	/* the length of the input vectors */
  if(rows != numIn)
    mexErrMsgTxt("Input vector length must equal number of input nodes.\n");

 /* Load Trainng Parameters  */

 /* Allocate Memory and Assign Pointers */
  allocReturnMatrices(plhs);	   /* Create Space for the return Arguments */
  assignMatrixPointers(plhs,prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */

 /* Copy initial weights into the training weight matrices */
  numWeights = numOut*numHid;
  numWeights = numHid * numIn * numTaps ;
/*

}
*/
   train();
}


/* calculate ---------------------------------------------------------------------*/

void train(void)
{ int epoch,i;

  fwd_pass();	
  back_prop();	
  weight_update();

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

  
}


/* Jacobian calculation */

void weight_update(void)
{
int i,j,t;
int tmp;
double muXdelta;
int numHidXnumIn = numHid*numIn;


    muXdelta =  deltaOut[0];
    for(j=0;j<numHid;j++)
      W2L[c2MAT(0,j,numOut)] = muXdelta * hidOut[j];
    B2L[0] = muXdelta;


  for(i=0;i<numHid;i++)
  {
    muXdelta =  deltaHid[i];
    for(t=0;t<numTaps;t++)
    {
      tmp =  - t;
      for(j=0;j<numIn;j++)
        W1L[c2MAT_tap(i,j,t,numHid,numHidXnumIn)] += muXdelta * 
                                             X[c2MAT(j,tmp,numIn)];
    }
    B1L[i] = muXdelta;
  }

}




/*----------------------------------------------------------------------*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numHid,(numTaps*numIn), mxREAL); /* creating W1L */
  plhs[1] = mxCreateDoubleMatrix(numHid,1, mxREAL);               /* creating B1L */
  plhs[2] = mxCreateDoubleMatrix(numOut,numHid, mxREAL);		/* creating W2L */
  plhs[3] = mxCreateDoubleMatrix(numOut,1, mxREAL);		/* creating B2L */
}

/*----------------------------------------------------------------------*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1R = mxGetPr(prhs[0]);
  B1R = mxGetPr(prhs[1]);
  W2R = mxGetPr(prhs[2]);
  B2R = mxGetPr(prhs[3]);
  X = mxGetPr(prhs[4]);

  W1L = mxGetPr(plhs[0]);
  B1L = mxGetPr(plhs[1]);
  W2L = mxGetPr(plhs[2]);
  B2L = mxGetPr(plhs[3]);
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

  deltaHid = (double *) mxCalloc (numHid , sizeof(double));
  if(deltaHid==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  deltaOut = (double *) mxCalloc (numOut , sizeof(double));
  if(deltaOut==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

}




































